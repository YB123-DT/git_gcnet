import unittest
from pathlib import Path


class CompletionTests(unittest.TestCase):
    def test_command_only_launches_seed66_flat_and_preserves_old_queue(self):
        from experiments.osram_mosei_flat_lr1e4_seed66_20261023.run import train_command
        command = train_command(Path('/fresh/run'), 'sha', 'GPU-good')
        self.assertEqual(command[command.index('--seed')+1], '66')
        self.assertEqual(command[command.index('--mode')+1], 'flat')
        self.assertIn('experiments.osram_mosei_gcnet_flat_lr1e4_20261021.worker', command)
        self.assertNotIn('--resume', command)
        self.assertIn('--resume', train_command(Path('/fresh/run'), 'sha', 'GPU-good', resume=True))


if __name__ == '__main__':
    unittest.main()
