import unittest

import torch
import torch.nn as nn

from model import GraphModel


class GraphModelBiLSTMAblationTests(unittest.TestCase):
    def _model(self, pre="bilstm", post="bilstm", **overrides):
        arguments = dict(
            base_model="LSTM",
            adim=2,
            tdim=2,
            vdim=2,
            D_e=4,
            graph_hidden_size=2,
            n_speakers=2,
            window_past=1,
            window_future=1,
            n_classes=6,
            dropout=0.0,
            time_attn=False,
            no_cuda=True,
            graph_conv_variant="original",
            pre_graph_context=pre,
            post_graph_context=post,
        )
        arguments.update(overrides)
        return GraphModel(**arguments)

    @staticmethod
    def _inputs():
        sequence_length, batch_size = 3, 2
        inputfeats = [torch.randn(sequence_length, batch_size, 6)]
        modality_mask = torch.ones(sequence_length, batch_size, 3)
        qmask = torch.tensor(
            [[0, 1, 0], [1, 0, 0]], dtype=torch.float32
        )
        umask = torch.tensor(
            [[1, 1, 1], [1, 1, 0]], dtype=torch.float32
        )
        return inputfeats, modality_mask, qmask, umask, [3, 2]

    def test_all_factorial_cells_share_modules_and_select_requested_modes(self):
        for pre_graph_context in ("bilstm", "linear"):
            for post_graph_context in ("bilstm", "linear"):
                with self.subTest(pre=pre_graph_context, post=post_graph_context):
                    model = self._model(pre_graph_context, post_graph_context)

                    self.assertIsInstance(model.lstm, nn.LSTM)
                    self.assertIsInstance(model.pre_graph_projection, nn.Linear)
                    self.assertEqual(model.pre_graph_context, pre_graph_context)
                    for branch in (
                        model.graph_net_temporal,
                        model.graph_net_speaker,
                    ):
                        self.assertIsInstance(branch.grufusion, nn.LSTM)
                        self.assertIsInstance(
                            branch.post_graph_projection, nn.Linear
                        )
                        self.assertEqual(
                            branch.post_graph_context, post_graph_context
                        )

    def test_invalid_context_modes_are_rejected(self):
        for keyword in ("pre_graph_context", "post_graph_context"):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(ValueError, keyword):
                    self._model(**{keyword: "gru"})

    def test_all_factorial_cells_have_finite_equal_shaped_outputs(self):
        expected_shapes = ((3, 2, 6), (3, 2, 6), (3, 2, 10))
        for pre_graph_context in ("bilstm", "linear"):
            for post_graph_context in ("bilstm", "linear"):
                with self.subTest(pre=pre_graph_context, post=post_graph_context):
                    torch.manual_seed(11)
                    model = self._model(pre_graph_context, post_graph_context)
                    outputs = model(*self._inputs())
                    tensors = (outputs[0], outputs[1][0], outputs[2])

                    self.assertEqual(
                        tuple(tuple(tensor.shape) for tensor in tensors),
                        expected_shapes,
                    )
                    for tensor in tensors:
                        self.assertTrue(torch.isfinite(tensor).all())

    def test_linear_replacements_are_utterance_local(self):
        model = self._model("linear", "linear")
        first = torch.randn(3, 2, 6)
        second = first.clone()
        second[2] = second[2] + 100.0
        pre_first = model.pre_graph_projection(first)
        pre_second = model.pre_graph_projection(second)
        self.assertTrue(torch.equal(pre_first[0], pre_second[0]))

        branch = model.graph_net_temporal
        first = torch.randn(3, 2, 10)
        second = first.clone()
        second[2] = second[2] - 100.0
        post_first = branch.post_graph_projection(first)
        post_second = branch.post_graph_projection(second)
        self.assertTrue(torch.equal(post_first[0], post_second[0]))

    def test_only_selected_factor_paths_receive_gradients(self):
        for pre_graph_context in ("bilstm", "linear"):
            for post_graph_context in ("bilstm", "linear"):
                with self.subTest(pre=pre_graph_context, post=post_graph_context):
                    torch.manual_seed(17)
                    model = self._model(pre_graph_context, post_graph_context)
                    outputs = model(*self._inputs())
                    sum(tensor.sum() for tensor in (
                        outputs[0], outputs[1][0], outputs[2]
                    )).backward()

                    pre_selected = (
                        model.lstm
                        if pre_graph_context == "bilstm"
                        else model.pre_graph_projection
                    )
                    pre_bypassed = (
                        model.pre_graph_projection
                        if pre_graph_context == "bilstm"
                        else model.lstm
                    )
                    self.assertTrue(
                        all(p.grad is not None for p in pre_selected.parameters())
                    )
                    self.assertTrue(
                        all(p.grad is None for p in pre_bypassed.parameters())
                    )

                    for branch in (
                        model.graph_net_temporal,
                        model.graph_net_speaker,
                    ):
                        post_selected = (
                            branch.grufusion
                            if post_graph_context == "bilstm"
                            else branch.post_graph_projection
                        )
                        post_bypassed = (
                            branch.post_graph_projection
                            if post_graph_context == "bilstm"
                            else branch.grufusion
                        )
                        self.assertTrue(
                            all(
                                p.grad is not None
                                for p in post_selected.parameters()
                            )
                        )
                        self.assertTrue(
                            all(
                                p.grad is None
                                for p in post_bypassed.parameters()
                            )
                        )

    def test_default_and_explicit_on_on_preserve_rng_parameters_and_outputs(self):
        seed = 23
        torch.manual_seed(seed)
        default = GraphModel(
            base_model="LSTM",
            adim=2,
            tdim=2,
            vdim=2,
            D_e=4,
            graph_hidden_size=2,
            n_speakers=2,
            window_past=1,
            window_future=1,
            n_classes=6,
            dropout=0.0,
            time_attn=False,
            no_cuda=True,
            graph_conv_variant="original",
        )
        default_rng_state = torch.get_rng_state().clone()

        torch.manual_seed(seed)
        explicit = self._model("bilstm", "bilstm")
        explicit_rng_state = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(default_rng_state, explicit_rng_state))

        default_parameters = dict(default.named_parameters())
        explicit_parameters = dict(explicit.named_parameters())
        official_names = {
            name for name in default_parameters
            if "_graph_projection" not in name
        }
        for name in official_names:
            self.assertTrue(
                torch.equal(default_parameters[name], explicit_parameters[name]),
                name,
            )

        default.eval()
        explicit.eval()
        torch.manual_seed(29)
        arguments = self._inputs()
        with torch.no_grad():
            expected = default(*arguments)
            actual = explicit(*arguments)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1][0], actual[1][0]))
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_all_factorial_cells_preserve_common_parameters_and_rng_state(self):
        seed = 31
        torch.manual_seed(seed)
        reference = self._model("bilstm", "bilstm")
        reference_rng_state = torch.get_rng_state().clone()
        reference_parameters = dict(reference.named_parameters())

        for pre_graph_context in ("bilstm", "linear"):
            for post_graph_context in ("bilstm", "linear"):
                with self.subTest(pre=pre_graph_context, post=post_graph_context):
                    torch.manual_seed(seed)
                    model = self._model(pre_graph_context, post_graph_context)
                    self.assertTrue(
                        torch.equal(reference_rng_state, torch.get_rng_state())
                    )
                    parameters = dict(model.named_parameters())
                    self.assertEqual(
                        set(reference_parameters), set(parameters)
                    )
                    for name, expected in reference_parameters.items():
                        self.assertTrue(torch.equal(expected, parameters[name]), name)

    def test_selected_path_parameter_counts_match_locked_original_contract(self):
        expected = {
            ("bilstm", "bilstm"): 34_140_166,
            ("linear", "bilstm"): 29_782_166,
            ("bilstm", "linear"): 15_110_166,
            ("linear", "linear"): 10_752_166,
        }
        for modes, expected_count in expected.items():
            with self.subTest(pre=modes[0], post=modes[1]):
                model = self._model(
                    *modes,
                    adim=1024,
                    tdim=1024,
                    vdim=512,
                    D_e=200,
                    graph_hidden_size=100,
                )
                self.assertEqual(
                    model.selected_path_parameter_count(), expected_count
                )


if __name__ == "__main__":
    unittest.main()
