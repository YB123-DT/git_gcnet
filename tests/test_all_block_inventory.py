"""Inventory must never manufacture scores, protocols or completed runs."""
import unittest

from experiments.osram_all_block_inventory_20261023.inventory import (
    normalized_metrics, protocol, extract_document, classify_status, embedded_runs, pick_mean,
)


class InventoryTests(unittest.TestCase):
    def test_metrics_units_and_ua_not_macro_f1(self):
        d=normalized_metrics(dict(accuracy=.6,weighted_f1=.57,macro_recall=.35,macro_f1=.32))
        for key,value in dict(ACC=60.,WF1=57.,UA=35.).items():
            self.assertAlmostEqual(d[key],value)
        for key,value in normalized_metrics(dict(ACC=60.,UA=35.,WF1=57.)).items():
            self.assertAlmostEqual(d[key],value)
        self.assertNotIn('UA',normalized_metrics(dict(macro_f1=.4)))

    def test_unknown_protocol_never_inferred_from_date(self):
        self.assertEqual(protocol({}, {}, {})[0],'unknown')
        self.assertEqual(protocol(dict(missing_mask_protocol='gcnet-random-mask-v1'),{}, {})[0],
                         'gcnet-random-mask-v1')
        self.assertEqual(protocol({},dict(identity=dict(source_hashes={
            'gcnet_missing_m3/train_gcnet.py':'abc'})),{'abc':dict(commit='old')})[0],
                         'conversation-mask-v2')

    def test_metric_rows_keep_rate_scope_and_source(self):
        d=dict(dataset='CMUMOSI',seed=66,epoch=30,status='running',
               evaluation={'0.7':{'router':dict(ACC=70.,WF1=69.)}})
        rows=extract_document(d,'/run/SUMMARY.json',{}, {})
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['rate'],'0.7')
        self.assertEqual(rows[0]['scope'],'rate')
        self.assertEqual(rows[0]['WF1'],69.)
        self.assertEqual(rows[0]['source_file'],'/run/SUMMARY.json')

    def test_no_score_for_config_or_runtime_only(self):
        self.assertFalse(extract_document(dict(dataset='MELD',epochs=50,
            learning_rate=.001),'/config.json',{},{}))
        self.assertFalse(extract_document(dict(status='preflight',epoch=None),
            '/SUMMARY.json',{},{}))

    def test_no_completed_claim_from_score_only_and_smoke_is_separate(self):
        self.assertEqual(classify_status({},True),'results_present_completion_unknown')
        self.assertEqual(classify_status(dict(status='complete',smoke_only=True),True),'smoke_complete')
        self.assertEqual(classify_status(dict(status='running',epoch=4),True),'running')

    def test_same_acc_and_ua_epoch_not_invented_from_two_selection_scopes(self):
        rows=extract_document(dict(aggregate=dict(testbest=dict(ACC=61.),
            paired=dict(UA=37.))),'/SUMMARY.json',{}, {})
        self.assertEqual(len(rows),2)
        self.assertFalse(any('ACC' in r and 'UA' in r for r in rows))

    def test_archived_runs_and_hpo_trials_survive_checkpoint_cleanup(self):
        cfg=dict(dataset='MELD',seed=66,learning_rate=1e-5,
                 missing_mask_protocol='gcnet-random-mask-v1')
        obj=dict(trials=[dict(config=cfg,number=16,state='COMPLETE',
            testbest=dict(mean=dict(ACC=61.,UA=36.)),
            paired=dict(on=dict(mean=dict(ACC=60.,UA=35.))))])
        rows=embedded_runs(obj,'/SUMMARY.json','test','local',{})
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['ACC'],61.)
        self.assertEqual(rows[0]['PAIRED_ACC'],60.)
        self.assertEqual(rows[0]['trial'],16)
        self.assertEqual(rows[0]['status'],'complete_declared')

    def test_nested_run_configuration_overrides_parent_identity(self):
        obj=dict(config=dict(dataset='MOSI',seed=66),
                 child=dict(config=dict(dataset='MOSEI',seed=67),accuracy=.8))
        row=extract_document(obj,'/SUMMARY.json',{}, {})[0]
        self.assertEqual((row['dataset'],row['seed']),('MOSEI',67))
        row=extract_document(dict(seed=68,accuracy=.8),'/metrics.json',
                             dict(dataset='MOSI',seed=66),{})[0]
        self.assertEqual(row['seed'],68)

    def test_selected_scores_are_not_replaced_by_final_random_retest(self):
        rates={f'{i/10:.1f}':.8 for i in range(8)}
        raw=dict(selected_accuracy_by_rate=rates,selected_weighted_f1_by_rate=rates,
                 test={r:dict(accuracy=.7,weighted_f1=.7,macro_recall=.4) for r in rates})
        self.assertEqual(pick_mean(raw),dict(ACC=80.,WF1=80.))
        self.assertNotIn('UA',pick_mean(raw))


if __name__=='__main__':unittest.main()
