"""
AWS EC2 client for the SN125 validator — the standardized-spec first-party
cloud provider.

Why AWS is in the provider set: Targon and Lambda sell whatever B200 box is
free, and the fixed wall-clock budget turns any box-to-box speed difference
straight into loss. EC2 lets the operator pin ONE exact machine shape — the
``p6-b200.48xlarge`` (8× B200 SXM 180 GB, 192 vCPU, 2 TiB RAM, 8× 3.8 TB NVMe,
3.2 Tbps EFA) — plus one exact image and one exact disk profile, and every
rental is bit-for-bit that spec (see :data:`AWS_STANDARD_SPEC`). As a
first-party cloud it is trusted for scoring (``cloud.SCORING_PROVIDERS``).

``AwsClient`` speaks the SAME duck-typed client interface that
``cloud.TargonOrchestrator`` drives (``ensure_ssh_key``, ``check_availability``,
``create_workload``, ``deploy_workload``, ``wait_running``,
``get_workload_state``, ``delete_workload``, ``list_workloads``), so the whole
orchestrator — spend cap, egress lockdown, detached runs, audit trail,
teardown — is reused unchanged:

    orch = TargonOrchestrator(client=AwsClient(), resource="b200-small", ...)

Plus the two optional orchestrator hooks:

* ``ssh_base_cmd(uid)`` — instances are reached directly at ``root@ip:22``.
  The Deep Learning AMI boots with SSH for ``ubuntu`` only and the remote
  commands in cloud.py assume root + /workspace, so ``wait_running``
  bootstraps root's authorized_keys and /workspace via sudo before reporting
  ready (same shape as the Lambda client).
* ``cost_per_hour(resource)`` — live on-demand pricing from the AWS Price List
  API (whole instance, all 8 GPUs), falling back to a pinned table.

THE ONE THING TO KNOW ABOUT AWS B200s: there is no 1×B200 shape. The canonical
``b200-small`` (1× B200) SKU therefore rents the WHOLE 8-GPU p6-b200.48xlarge
and the eval runs on GPU 0 (``training.py`` pins CUDA_VISIBLE_DEVICES to the
first device). Cost accounting is honest — ``cost_per_hour`` reports the full
instance price (~$114/h on-demand) — so one 20h eval is ~$2,300 against the
``DAILY_SPEND_LIMIT``. Never report a per-GPU fraction here: the spend cap is
a real-money guard and must see what the account is actually billed.

Resource naming: canonical Targon-style SKUs translate to instance types
(``b200-small`` / ``b200-xlarge`` → ``p6-b200.48xlarge``, see SKU_INSTANCE_MAP);
raw EC2 instance type names (``p6-b200.48xlarge``, ``t3.small``) pass through.
``is_b200_sku`` keeps working (the type name contains "b200").

Capacity model: EC2 has no "how many are free" endpoint — a RunInstances
either succeeds or fails with InsufficientInstanceCapacity. So
``check_availability`` counts the (region, AZ) pairs where the type is
OFFERED and the account's vCPU QUOTA for that family has room for one more
box (the two failure modes that are knowable up front); ``create_workload``
then walks those AZs and treats capacity/quota errors as "try the next one".
A fresh account has a P-family quota of 0 vCPU — see scripts/aws_bootstrap.py
for the quota-increase request.

Auto-termination: EC2 has NO platform-side kill timer. The backstops are the
orchestrator's teardown + startup orphan sweep (terminates every instance
tagged ``sn125:managed``), and the periodic staleness sweep, which here gets a
REAL created_at (LaunchTime) rather than a locally stamped one.

Credentials (in precedence order): the standard boto3 chain — AWS_ACCESS_KEY_ID
/ AWS_SECRET_ACCESS_KEY env, AWS_PROFILE, shared credentials file, instance
role — then the operator's AWS_ROOT_ACCESS_KEY / AWS_ROOT_ACCESS_SECRET pair
as a loudly-warned last resort (root keys should only bootstrap the scoped
``sn125-validator`` IAM user; scripts/aws_bootstrap.py does that). NEVER
hardcode keys in this file.

Dependency: boto3 (already required for R2 publishing).
"""
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .cloud import SSH_CONNECT_TIMEOUT, SSH_KEY_DIR, SSH_KEY_NAME
from .env import load_dotenv

log = logging.getLogger("sn125.cloud_aws")

AWS_SSH_KEY_PATH = SSH_KEY_DIR / "aws_id_ed25519"
AWS_SECURITY_GROUP_NAME = "sn125-validator-ssh"
MANAGED_TAG = "sn125:managed"

REGIONS_ENV = "SN125_AWS_REGIONS"
SSH_CIDR_ENV = "SN125_AWS_SSH_CIDR"
AMI_SSM_PARAMETER_ENV = "SN125_AWS_AMI_SSM_PARAMETER"
AMI_ENV = "SN125_AWS_AMI"
ROOT_GIB_ENV = "SN125_AWS_ROOT_GIB"
STARTUP_TIMEOUT_ENV = "SN125_AWS_STARTUP_TIMEOUT"

DEFAULT_REGIONS = ("us-east-2", "us-west-2", "us-east-1")

AWS_STARTUP_TIMEOUT = int(os.environ.get(STARTUP_TIMEOUT_ENV, "1500"))

AWS_STANDARD_SPEC: dict = {
    "instance_type": "p6-b200.48xlarge",
    "ami_ssm_parameter": ("/aws/service/deeplearning/ami/x86_64/"
                          "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"),
    "login_user": "ubuntu",
    "root_volume_gib": 500,
    "root_volume_type": "gp3",
    "root_iops": 3000,
    "root_throughput_mbps": 500,
    "ebs_optimized": True,
    "tenancy": "default",
    "market": "on-demand",
    "imds": "v2-required",
    "shutdown_behavior": "terminate",
}

SKU_INSTANCE_MAP = {
    "b200-small": "p6-b200.48xlarge",
    "b200-xlarge": "p6-b200.48xlarge",
    "h100-xlarge": "p5.48xlarge",
    "h200-xlarge": "p5en.48xlarge",
    "cpu-small": "t3.small",
    "cpu-medium": "c7i.xlarge",
}

_QUOTA_CODES = {
    "p": "L-417A185B",
    "g": "L-DB2E81BA",
    "vt": "L-DB2E81BA",
}
_STANDARD_QUOTA_CODE = "L-1216C47A"

ONDEMAND_USD_FALLBACK = {
    "p6-b200.48xlarge": 113.93,
    "p5.48xlarge": 55.04,
    "p5en.48xlarge": 64.0,
    "t3.small": 0.0208,
    "c7i.xlarge": 0.1785,
}

_STATUS_MAP = {
    "pending": "PROVISIONING",
    "running": "RUNNING",
    "shutting-down": "DELETED",
    "terminated": "DELETED",
    "stopping": "FAILED",
    "stopped": "FAILED",
}
_TERMINAL_STATUSES = frozenset({"FAILED", "DELETED"})

_RETRYABLE_LAUNCH_ERRORS = frozenset({
    "InsufficientInstanceCapacity", "Unsupported", "VcpuLimitExceeded",
    "InstanceLimitExceeded", "InsufficientCapacityOnHost",
    "PendingVerification", "UnsupportedOperation",
})

_ROOT_BOOTSTRAP = (
    "sudo install -d -m 700 -o root -g root /root/.ssh && "
    "sudo install -m 600 -o root -g root ~/.ssh/authorized_keys "
    "/root/.ssh/authorized_keys && "
    "sudo mkdir -p /workspace && echo bootstrapped"
)


def resolve_resource(resource: str) -> str:
    """Translate a resource string into an EC2 instance type name.

    Canonical Targon-style SKUs go through SKU_INSTANCE_MAP; raw EC2 type
    names (``p6-b200.48xlarge``) pass through unchanged."""
    if resource in SKU_INSTANCE_MAP:
        return SKU_INSTANCE_MAP[resource]
    return resource.strip()


def instance_family(instance_type: str) -> str:
    """Family letters of an instance type: ``p6-b200.48xlarge`` → ``p``,
    ``c7i.xlarge`` → ``c``, ``vt1.3xlarge`` → ``vt``."""
    head = instance_type.split(".", 1)[0]
    letters = ""
    for ch in head:
        if ch.isalpha():
            letters += ch
        else:
            break
    return letters.lower()


def quota_code_for(instance_type: str) -> str:
    """Service Quotas code governing on-demand vCPUs of this type's family."""
    return _QUOTA_CODES.get(instance_family(instance_type), _STANDARD_QUOTA_CODE)


def configured_regions() -> list[str]:
    """Ordered region list (SN125_AWS_REGIONS, comma-separated) or the default."""
    raw = os.environ.get(REGIONS_ENV, "")
    regions = [r.strip() for r in raw.split(",") if r.strip()]
    return regions or list(DEFAULT_REGIONS)


def _pinned_ami(region: str) -> str:
    """Operator-pinned AMI for ``region`` from SN125_AWS_AMI: either a single
    ``ami-...`` (same id for every region — only valid for one region, but
    honored) or ``region=ami-...`` pairs. "" when nothing is pinned."""
    raw = os.environ.get(AMI_ENV, "").strip()
    if not raw:
        return ""
    if "=" not in raw:
        return raw
    for tok in raw.split(","):
        if "=" in tok:
            r, _, ami = tok.partition("=")
            if r.strip() == region:
                return ami.strip()
    return ""


def _default_session():
    """boto3 session from the standard credential chain, falling back to the
    operator's root key pair with a loud warning."""
    import boto3
    load_dotenv()
    session = boto3.session.Session()
    if session.get_credentials() is not None:
        return session
    root_key = os.environ.get("AWS_ROOT_ACCESS_KEY", "").strip()
    root_secret = os.environ.get("AWS_ROOT_ACCESS_SECRET", "").strip()
    if root_key and root_secret:
        log.warning(
            "AWS: no scoped credentials found — falling back to the ROOT access "
            "key. Run scripts/aws_bootstrap.py to create the scoped sn125-validator "
            "IAM user and set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY.")
        return boto3.session.Session(aws_access_key_id=root_key,
                                     aws_secret_access_key=root_secret)
    raise RuntimeError(
        "No AWS credentials found. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
        "(scoped IAM user — see scripts/aws_bootstrap.py) or AWS_PROFILE.")


def _error_code(exc: Exception) -> str:
    """botocore ClientError code, or "" for anything else."""
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        return str((resp.get("Error") or {}).get("Code") or "")
    return ""


class AwsClient:
    """EC2 provider client behind the orchestrator's duck-typed seam.

    ``client_factory(service, region)`` returns a boto3-shaped client; tests
    inject fakes here so no test needs credentials, network, or moto."""

    provider_name = "aws"

    def __init__(self, regions: list[str] | None = None, key_path: str = "",
                 client_factory: Callable[[str, str], object] | None = None,
                 spec: dict | None = None):
        self.regions = list(regions) if regions else configured_regions()
        if not self.regions:
            raise RuntimeError(f"no AWS regions configured ({REGIONS_ENV})")
        self.spec = dict(AWS_STANDARD_SPEC, **(spec or {}))
        self.key_path = Path(key_path) if key_path else AWS_SSH_KEY_PATH
        self._session = None
        self._factory = client_factory
        self._clients: dict[tuple[str, str], object] = {}
        self._region_of: dict[str, str] = {}
        self._ssh_ip: dict[str, str] = {}
        self._key_names: dict[str, str] = {}
        self._sg_ids: dict[str, str] = {}
        self._ami_cache: dict[str, dict] = {}
        self._offer_cache: dict[tuple[str, str], tuple[list[str], float]] = {}
        self._quota_cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._vcpu_cache: dict[str, int] = {}
        self._price_cache: dict[str, tuple[float, float]] = {}
        self._cache_ttl_s = 300.0

    def _client(self, service: str, region: str):
        key = (service, region)
        if key not in self._clients:
            if self._factory is not None:
                self._clients[key] = self._factory(service, region)
            else:
                if self._session is None:
                    self._session = _default_session()
                self._clients[key] = self._session.client(service, region_name=region)
        return self._clients[key]

    def _ec2(self, region: str):
        return self._client("ec2", region)

    def _fresh(self, entry, ttl: float | None = None) -> bool:
        return bool(entry) and (time.time() - entry[1]) < (ttl or self._cache_ttl_s)

    def _vcpus(self, region: str, itype: str) -> int:
        if itype not in self._vcpu_cache:
            resp = self._ec2(region).describe_instance_types(InstanceTypes=[itype])
            types = resp.get("InstanceTypes") or []
            if not types:
                raise RuntimeError(f"AWS: unknown instance type {itype!r}")
            self._vcpu_cache[itype] = int(types[0]["VCpuInfo"]["DefaultVCpus"])
        return self._vcpu_cache[itype]

    def _offered_azs(self, region: str, itype: str) -> list[str]:
        """AZ names in ``region`` that offer ``itype`` (cached ~5 min)."""
        key = (region, itype)
        if self._fresh(self._offer_cache.get(key)):
            return list(self._offer_cache[key][0])
        try:
            resp = self._ec2(region).describe_instance_type_offerings(
                LocationType="availability-zone",
                Filters=[{"Name": "instance-type", "Values": [itype]}])
            azs = sorted(o["Location"] for o in resp.get("InstanceTypeOfferings") or [])
        except Exception as e:
            log.warning(f"AWS {region}: offerings lookup for {itype} failed: {e}")
            azs = []
        self._offer_cache[key] = (azs, time.time())
        return list(azs)

    def _quota_vcpus(self, region: str, itype: str) -> float:
        """Applied on-demand vCPU quota for this type's family (cached ~5 min).
        Unknown/unreadable quota reads as 'plenty' so a Service Quotas outage
        degrades to the RunInstances-error path rather than a false 'dry'."""
        code = quota_code_for(itype)
        key = (region, code)
        if self._fresh(self._quota_cache.get(key)):
            return self._quota_cache[key][0]
        try:
            q = self._client("service-quotas", region).get_service_quota(
                ServiceCode="ec2", QuotaCode=code)
            value = float(q["Quota"]["Value"])
        except Exception as e:
            log.warning(f"AWS {region}: quota {code} lookup failed: {e}")
            value = float("inf")
        self._quota_cache[key] = (value, time.time())
        return value

    def _vcpus_in_use(self, region: str, itype: str) -> int:
        """vCPUs currently consumed in ``region`` by live instances of the same
        quota family (they count against the same cap as the next launch)."""
        fam = instance_family(itype)
        try:
            resp = self._ec2(region).describe_instances(Filters=[
                {"Name": "instance-state-name", "Values": ["pending", "running"]}])
        except Exception as e:
            log.warning(f"AWS {region}: describe_instances failed: {e}")
            return 0
        used = 0
        for res in resp.get("Reservations") or []:
            for inst in res.get("Instances") or []:
                it = inst.get("InstanceType", "")
                if quota_code_for(it) == quota_code_for(itype) or instance_family(it) == fam:
                    try:
                        used += self._vcpus(region, it)
                    except Exception:
                        pass
        return used

    def _launchable_azs(self, resource_name: str) -> list[tuple[str, str]]:
        """(region, az) pairs, in region preference order, where the type is
        offered AND the family quota has room for one more instance."""
        itype = resolve_resource(resource_name)
        out: list[tuple[str, str]] = []
        for region in self.regions:
            azs = self._offered_azs(region, itype)
            if not azs:
                continue
            try:
                need = self._vcpus(region, itype)
            except Exception as e:
                log.warning(f"AWS {region}: {e}")
                continue
            quota = self._quota_vcpus(region, itype)
            if quota < need:
                log.info(f"AWS {region}: {itype} needs {need} vCPU, on-demand quota "
                         f"{quota_code_for(itype)} is {quota:.0f} — request an increase "
                         f"(scripts/aws_bootstrap.py --request-quota)")
                continue
            if quota != float("inf") and quota - self._vcpus_in_use(region, itype) < need:
                log.info(f"AWS {region}: {itype} quota {quota:.0f} vCPU fully in use")
                continue
            out.extend((region, az) for az in azs)
        return out

    def check_availability(self, resource_name: str) -> int:
        """Number of (region, AZ) slots that could serve one rental right now
        (the orchestrator only needs >= 1). Real capacity is only knowable at
        RunInstances time — this rules out the two knowable blockers (type not
        offered, quota exhausted)."""
        return len(self._launchable_azs(resource_name))

    def cost_per_hour(self, resource_name: str) -> float:
        """Live on-demand $/h for the WHOLE instance (Price List API, Linux,
        shared tenancy, first configured region), falling back to the pinned
        table. Never a per-GPU fraction — see module docstring."""
        itype = resolve_resource(resource_name)
        if self._fresh(self._price_cache.get(itype), ttl=3600.0):
            return self._price_cache[itype][0]
        price = self._live_price(itype)
        if price is None:
            price = ONDEMAND_USD_FALLBACK.get(itype)
        if price is None:
            log.warning(f"AWS: no price for {itype!r}; using pessimistic $120/h")
            price = 120.0
        self._price_cache[itype] = (float(price), time.time())
        return float(price)

    def _live_price(self, itype: str) -> float | None:
        import json
        try:
            pricing = self._client("pricing", "us-east-1")
            resp = pricing.get_products(ServiceCode="AmazonEC2", MaxResults=5, Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": itype},
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": self.regions[0]},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            ])
            best = None
            for raw in resp.get("PriceList") or []:
                doc = json.loads(raw) if isinstance(raw, str) else raw
                for term in (doc.get("terms") or {}).get("OnDemand", {}).values():
                    for dim in (term.get("priceDimensions") or {}).values():
                        usd = float((dim.get("pricePerUnit") or {}).get("USD", 0) or 0)
                        if usd > 0 and (best is None or usd > best):
                            best = usd
            return best
        except Exception as e:
            log.warning(f"AWS: live price lookup failed for {itype!r}: {e}")
            return None

    @staticmethod
    def _key_material(pubkey: str) -> str:
        """type + base64 blob of an OpenSSH public key (comment stripped)."""
        return " ".join((pubkey or "").split()[:2])

    def _local_pubkey(self) -> str:
        SSH_KEY_DIR.mkdir(parents=True, exist_ok=True)
        if not self.key_path.exists():
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", str(self.key_path),
                 "-N", "", "-C", SSH_KEY_NAME],
                check=True, capture_output=True)
            log.info(f"Generated SSH keypair at {self.key_path}")
        return self.key_path.with_suffix(".pub").read_text().strip()

    def _ensure_key_in_region(self, region: str, pubkey: str) -> str:
        """Import the local public key into ``region`` (idempotent by material)."""
        if region in self._key_names:
            return self._key_names[region]
        ec2 = self._ec2(region)
        material = self._key_material(pubkey)
        taken = set()
        for kp in ec2.describe_key_pairs(IncludePublicKey=True).get("KeyPairs") or []:
            taken.add(kp.get("KeyName", ""))
            if self._key_material(kp.get("PublicKey", "")) == material:
                self._key_names[region] = kp["KeyName"]
                return kp["KeyName"]
        name = SSH_KEY_NAME
        if name in taken:
            name = f"{SSH_KEY_NAME}-{material[-8:].replace('/', '_').replace('+', '-')}"
        ec2.import_key_pair(KeyName=name, PublicKeyMaterial=pubkey.encode(),
                            TagSpecifications=[{"ResourceType": "key-pair",
                                                "Tags": [{"Key": MANAGED_TAG, "Value": "1"}]}])
        log.info(f"AWS {region}: imported SSH key pair '{name}'")
        self._key_names[region] = name
        return name

    def ensure_ssh_key(self) -> str:
        """Generate the local keypair if needed and import it into EVERY
        configured region. Returns the key name in the first region (launches
        look the per-region name up themselves)."""
        pubkey = self._local_pubkey()
        first = ""
        for region in self.regions:
            try:
                name = self._ensure_key_in_region(region, pubkey)
            except Exception as e:
                log.warning(f"AWS {region}: key import failed: {e}")
                continue
            first = first or name
        if not first:
            raise RuntimeError("AWS: could not register the SSH key in any configured region")
        return first

    def _ensure_security_group(self, region: str) -> str:
        """Find-or-create the SSH-only security group in the default VPC."""
        if region in self._sg_ids:
            return self._sg_ids[region]
        ec2 = self._ec2(region)
        found = ec2.describe_security_groups(Filters=[
            {"Name": "group-name", "Values": [AWS_SECURITY_GROUP_NAME]}]).get("SecurityGroups") or []
        if found:
            sg_id = found[0]["GroupId"]
        else:
            vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}]).get("Vpcs") or []
            if not vpcs:
                raise RuntimeError(f"AWS {region}: no default VPC (create one or add VPC support)")
            sg_id = ec2.create_security_group(
                GroupName=AWS_SECURITY_GROUP_NAME,
                Description="SN125 validator eval boxes: SSH only",
                VpcId=vpcs[0]["VpcId"],
                TagSpecifications=[{"ResourceType": "security-group",
                                    "Tags": [{"Key": MANAGED_TAG, "Value": "1"}]}])["GroupId"]
            log.info(f"AWS {region}: created security group {sg_id}")
        cidr = os.environ.get(SSH_CIDR_ENV, "").strip() or "0.0.0.0/0"
        try:
            ec2.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                "IpRanges": [{"CidrIp": cidr, "Description": "sn125 validator ssh"}]}])
        except Exception as e:
            if _error_code(e) != "InvalidPermission.Duplicate":
                raise
        self._sg_ids[region] = sg_id
        return sg_id

    def _resolve_ami(self, region: str) -> dict:
        """The standardized image for ``region``: {"ami", "root_device", "min_gib"}.
        SN125_AWS_AMI pins an id; otherwise the spec's public SSM parameter is
        resolved (same DLAMI build in every region)."""
        if region in self._ami_cache:
            return self._ami_cache[region]
        ami = _pinned_ami(region)
        if not ami:
            param = os.environ.get(AMI_SSM_PARAMETER_ENV, "").strip() or self.spec["ami_ssm_parameter"]
            ami = self._client("ssm", region).get_parameter(Name=param)["Parameter"]["Value"]
        images = self._ec2(region).describe_images(ImageIds=[ami]).get("Images") or []
        if not images:
            raise RuntimeError(f"AWS {region}: AMI {ami} not found")
        img = images[0]
        root_dev = img.get("RootDeviceName", "/dev/sda1")
        min_gib = 8
        for bdm in img.get("BlockDeviceMappings") or []:
            if bdm.get("DeviceName") == root_dev:
                min_gib = int((bdm.get("Ebs") or {}).get("VolumeSize") or min_gib)
        info = {"ami": ami, "root_device": root_dev, "min_gib": min_gib,
                "name": img.get("Name", "")}
        self._ami_cache[region] = info
        return info

    def _root_volume_gib(self, ami_min: int) -> int:
        want = int(os.environ.get(ROOT_GIB_ENV, "") or self.spec["root_volume_gib"])
        return max(want, ami_min)

    def _subnet_for(self, region: str, az: str) -> str:
        subs = self._ec2(region).describe_subnets(Filters=[
            {"Name": "default-for-az", "Values": ["true"]},
            {"Name": "availability-zone", "Values": [az]}]).get("Subnets") or []
        if not subs:
            raise RuntimeError(f"AWS {region}: no default subnet in {az}")
        return subs[0]["SubnetId"]

    def create_workload(self, name: str, resource_name: str, ssh_key_uid: str,
                        image: str = "", envs: list[dict] = None,
                        args: list[str] = None) -> dict:
        """Launch exactly one standardized instance. Returns {"uid": instance_id, ...}.

        ``image``/``envs``/``args`` are part of the Targon-shaped interface;
        the image is pinned by the spec and eval env vars travel with the
        detached run command, so they are intentionally ignored. ``ssh_key_uid``
        is advisory — the per-region key name from ensure_ssh_key() is used."""
        itype = resolve_resource(resource_name)
        slots = self._launchable_azs(resource_name)
        if not slots:
            raise RuntimeError(f"no AWS capacity for {resource_name!r} ({itype}): "
                               f"not offered or vCPU quota exhausted in {self.regions}")
        if "b200" in itype and resource_name != "b200-xlarge":
            log.warning(f"AWS has no 1×B200 shape: {resource_name!r} rents the whole "
                        f"8-GPU {itype} (~${self.cost_per_hour(resource_name):.2f}/h); "
                        f"the eval runs on GPU 0.")
        pubkey = self._local_pubkey()
        last_err: Exception | None = None
        tried: list[str] = []
        for region, az in slots:
            try:
                key_name = self._ensure_key_in_region(region, pubkey)
                sg_id = self._ensure_security_group(region)
                ami = self._resolve_ami(region)
                subnet = self._subnet_for(region, az)
            except Exception as e:
                last_err = e
                log.warning(f"AWS {region}/{az}: pre-launch setup failed: {e}")
                continue
            tags = [
                {"Key": "Name", "Value": name},
                {"Key": MANAGED_TAG, "Value": "1"},
                {"Key": "sn125:provider", "Value": "aws"},
                {"Key": "sn125:resource", "Value": resource_name},
                {"Key": "sn125:ami", "Value": ami["ami"]},
            ]
            params = {
                "ImageId": ami["ami"],
                "InstanceType": itype,
                "KeyName": key_name,
                "MinCount": 1, "MaxCount": 1,
                "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": subnet,
                                       "Groups": [sg_id],
                                       "AssociatePublicIpAddress": True}],
                "BlockDeviceMappings": [{
                    "DeviceName": ami["root_device"],
                    "Ebs": {"VolumeSize": self._root_volume_gib(ami["min_gib"]),
                            "VolumeType": self.spec["root_volume_type"],
                            "Iops": self.spec["root_iops"],
                            "Throughput": self.spec["root_throughput_mbps"],
                            "DeleteOnTermination": True}}],
                "EbsOptimized": bool(self.spec["ebs_optimized"]),
                "InstanceInitiatedShutdownBehavior": self.spec["shutdown_behavior"],
                "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
                "Placement": {"AvailabilityZone": az, "Tenancy": self.spec["tenancy"]},
                "TagSpecifications": [
                    {"ResourceType": "instance", "Tags": tags},
                    {"ResourceType": "volume", "Tags": tags},
                ],
            }
            tried.append(f"{region}/{az}")
            try:
                resp = self._ec2(region).run_instances(**params)
            except Exception as e:
                code = _error_code(e)
                last_err = e
                if code in _RETRYABLE_LAUNCH_ERRORS:
                    log.warning(f"AWS {region}/{az}: {code} for {itype} — trying next slot")
                    continue
                raise RuntimeError(f"AWS RunInstances {itype} in {region}/{az} failed: {e}") from e
            insts = resp.get("Instances") or []
            if not insts:
                last_err = RuntimeError(f"RunInstances returned no instances: {resp}")
                continue
            iid = str(insts[0]["InstanceId"])
            self._region_of[iid] = region
            log.info(f"AWS: launched {iid} ('{name}', {itype}, {region}/{az}, "
                     f"{ami['ami']} {ami['name']!r})")
            return {"uid": iid, "name": name, "region": region, "az": az,
                    "instance_type": itype, "ami": ami["ami"], "ami_name": ami["name"],
                    "spec": dict(self.spec)}
        raise RuntimeError(f"AWS launch failed for {resource_name!r} after {tried}: {last_err}")

    def deploy_workload(self, wrk_uid: str) -> dict:
        """No-op: RunInstances both registers and boots the instance."""
        return {}

    def _find_region(self, wrk_uid: str) -> str:
        """Region owning ``wrk_uid`` — from the launch map, else by scanning."""
        region = self._region_of.get(str(wrk_uid))
        if region:
            return region
        for r in self.regions:
            try:
                resp = self._ec2(r).describe_instances(InstanceIds=[str(wrk_uid)])
            except Exception:
                continue
            if any(res.get("Instances") for res in resp.get("Reservations") or []):
                self._region_of[str(wrk_uid)] = r
                return r
        return ""

    def _describe(self, wrk_uid: str) -> dict:
        region = self._find_region(wrk_uid)
        if not region:
            return {}
        try:
            resp = self._ec2(region).describe_instances(InstanceIds=[str(wrk_uid)])
        except Exception as e:
            if _error_code(e) in ("InvalidInstanceID.NotFound", "InvalidInstanceID.Malformed"):
                return {}
            raise
        for res in resp.get("Reservations") or []:
            for inst in res.get("Instances") or []:
                return inst
        return {}

    def get_workload_state(self, wrk_uid: str) -> dict:
        """Current instance state in the Targon-shaped {"status": ...} form."""
        inst = self._describe(wrk_uid)
        if not inst:
            return {"status": "DELETED"}
        ip = inst.get("PublicIpAddress") or ""
        if ip:
            self._ssh_ip[str(wrk_uid)] = ip
        raw = str((inst.get("State") or {}).get("Name", "")).lower()
        name = next((t["Value"] for t in inst.get("Tags") or [] if t.get("Key") == "Name"), "")
        return {"status": _STATUS_MAP.get(raw, raw.upper() or "UNKNOWN"),
                "name": name, "ip": ip,
                "instance_type": inst.get("InstanceType", ""),
                "region": self._region_of.get(str(wrk_uid), "")}

    def wait_running(self, wrk_uid: str, timeout: int = AWS_STARTUP_TIMEOUT,
                     poll_interval: int = 15) -> bool:
        """Poll until the instance is running, then bootstrap root SSH +
        /workspace (see _ROOT_BOOTSTRAP) so the box serves the same root@
        contract as Targon/Lium containers. False on timeout or terminal."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.get_workload_state(wrk_uid).get("status", "UNKNOWN")
            if status == "RUNNING":
                return self._bootstrap_root(wrk_uid, deadline)
            if status in _TERMINAL_STATUSES:
                log.error(f"AWS instance {wrk_uid} entered terminal state: {status}")
                return False
            time.sleep(poll_interval)
        log.error(f"AWS instance {wrk_uid} did not reach running within {timeout}s")
        return False

    def _bootstrap_root(self, wrk_uid: str, deadline: float) -> bool:
        ip = self._ssh_ip.get(str(wrk_uid), "")
        if not ip:
            log.error(f"AWS instance {wrk_uid} running but has no public ip")
            return False
        user = self.spec["login_user"]
        while time.time() < deadline:
            r = subprocess.run(self._ssh_cmd_for(f"{user}@{ip}") + [_ROOT_BOOTSTRAP],
                               capture_output=True, timeout=180)
            if r.returncode == 0 and b"bootstrapped" in r.stdout:
                v = subprocess.run(self._ssh_cmd_for(f"root@{ip}") + ["echo rootok"],
                                   capture_output=True, timeout=120)
                if v.returncode == 0 and b"rootok" in v.stdout:
                    log.info(f"AWS instance {wrk_uid} root-bootstrapped at {ip}")
                    return True
                log.warning(f"root SSH verify failed on {wrk_uid}: "
                            f"{v.stderr.decode(errors='replace')[-200:]}")
            time.sleep(10)
        log.error(f"AWS instance {wrk_uid} bootstrap did not complete in time")
        return False

    def delete_workload(self, wrk_uid: str) -> None:
        region = self._find_region(wrk_uid)
        if not region:
            log.warning(f"AWS: delete_workload({wrk_uid}): instance not found in {self.regions}")
            return
        try:
            self._ec2(region).terminate_instances(InstanceIds=[str(wrk_uid)])
            log.info(f"AWS {region}: terminate requested for {wrk_uid}")
        except Exception as e:
            if _error_code(e) not in ("InvalidInstanceID.NotFound",):
                log.warning(f"Failed to terminate AWS instance {wrk_uid}: {e}")

    def list_workloads(self, status: str = None) -> list[dict]:
        """sn125-managed instances across all configured regions in the Targon
        workload shape (name, uid, created_at, state.status). created_at is the
        real LaunchTime, so the periodic staleness sweep works for instances
        launched by a previous validator process too."""
        out = []
        for region in self.regions:
            try:
                resp = self._ec2(region).describe_instances(Filters=[
                    {"Name": f"tag:{MANAGED_TAG}", "Values": ["1"]},
                    {"Name": "instance-state-name",
                     "Values": ["pending", "running", "stopping", "stopped", "shutting-down"]}])
            except Exception as e:
                log.warning(f"AWS {region}: list instances failed: {e}")
                continue
            for res in resp.get("Reservations") or []:
                for inst in res.get("Instances") or []:
                    iid = str(inst.get("InstanceId", ""))
                    self._region_of[iid] = region
                    st = _STATUS_MAP.get(str((inst.get("State") or {}).get("Name", "")).lower(), "UNKNOWN")
                    if status and st != status.upper():
                        continue
                    name = next((t["Value"] for t in inst.get("Tags") or [] if t.get("Key") == "Name"), "")
                    lt = inst.get("LaunchTime")
                    if isinstance(lt, datetime):
                        created = lt.astimezone(timezone.utc).isoformat()
                    else:
                        created = str(lt or "")
                    out.append({"name": name, "uid": iid, "created_at": created,
                                "state": {"status": st}, "region": region})
        return out

    def _ssh_cmd_for(self, user_host: str) -> list[str]:
        return ["ssh", "-i", str(self.key_path),
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}",
                user_host]

    def ssh_base_cmd(self, wrk_uid: str) -> list[str]:
        """Base SSH command for an instance — root@public-ip:22 with the
        imported private key (root access set up by wait_running)."""
        ip = self._ssh_ip.get(str(wrk_uid))
        if not ip:
            ip = self.get_workload_state(wrk_uid).get("ip", "")
        if not ip:
            raise RuntimeError(f"no SSH connect info for AWS instance {wrk_uid}")
        return self._ssh_cmd_for(f"root@{ip}")
