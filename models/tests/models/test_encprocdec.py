#  import pickle as pkl
#
#  import torch
#
#  from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
#
#  base = "/home/masc/mllam/anemoi-core/experiment/state/"
#
#
#  def test_encprocdec_instantiate():
#      torch.manual_seed(42)
#      batch = pkl.load(open(base + "x.pkl", "rb"))
#      graph = pkl.load(open(base + "graph.pkl", "rb"))
#      model_config = pkl.load(open(base + "model_config.pkl", "rb"))
#      data_indices = pkl.load(open(base + "data_indices.pkl", "rb"))
#      statistics = pkl.load(open(base + "statistics.pkl", "rb"))
#      truncation_data = {}
#
#      model = AnemoiModelEncProcDec(
#          model_config=model_config,
#          statistics=statistics,
#          graph_data=graph,
#          data_indices=data_indices,
#          truncation_data=truncation_data,
#      )
#
#      out = model(batch)
#      assert out.sum() == -1553.62158203125
#
#
#  def test_encprocdec_instantiate_with_residual():
#      torch.manual_seed(42)
#      batch = pkl.load(open(base + "x.pkl", "rb"))
#      graph = pkl.load(open(base + "graph.pkl", "rb"))
#      model_config = pkl.load(open(base + "model_config.pkl", "rb"))
#      data_indices = pkl.load(open(base + "data_indices.pkl", "rb"))
#      statistics = pkl.load(open(base + "statistics.pkl", "rb"))
#      truncation_data = {}
#
#      model_config.model.residual = {}
#
#      model = AnemoiModelEncProcDec(
#          model_config=model_config,
#          statistics=statistics,
#          graph_data=graph,
#          data_indices=data_indices,
#          truncation_data=truncation_data,
#      )
#
#      out = model(batch)
#      assert out.sum() == -1553.62158203125
