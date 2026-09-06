"""SN125 round state machine (DESIGN.md §8 commit gate + §2.4 pause gates).

round_fsm : batched-round lifecycle commit -> reveal -> evaluate -> publish,
            wired to payments.registry for the credit gate.

(Named ``roundsm`` because ``sn125/rounds/`` is a pre-existing JSON data
directory consumed by rolling_baseline.py / dashboard / neuron.py.)
"""
