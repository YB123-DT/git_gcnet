import unittest
from experiments.osram_context_audit_20261003.analyze import classify,stats


class ContextAuditTests(unittest.TestCase):
    def test_mse_and_polarity_are_distinct(self):
        g,effect,transition=classify(1.,.9,2.)
        self.assertLess(g,0)
        self.assertEqual((effect,transition),('harm','both_correct'))

    def test_corrections_neutral_and_zero(self):
        self.assertEqual(classify(1.,-.5,.5)[2],'wrong_to_correct')
        self.assertEqual(classify(-1.,-.5,.5)[2],'correct_to_wrong')
        self.assertEqual(classify(0.,-.5,.5)[2],'neutral_excluded')
        self.assertEqual(classify(1.,.2,.2)[1],'near_zero')


if __name__=='__main__':unittest.main()
