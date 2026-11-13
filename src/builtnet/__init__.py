"""builtnet: register-style network construction rules, synthetic populations and their theory.

Main entry points:
  builtnet.cards         construction cards (machine-readable rule descriptions) -> RuleSet
  builtnet.rulesets      RuleSet, presets, build(world, ruleset, seed) -> MultilayerNetwork
  builtnet.tables        world_from_tables(...): apply any rule-set to real membership tables
  builtnet.world         synthetic populations (WorldConfig, generate)
  builtnet.descriptors   position descriptors (degree, clustering, excess closure, multiplex share)
  builtnet.measures      construction share and group measures
  builtnet.theory        expected layer degree and clustering from group sizes and density
"""

from .network import MultilayerNetwork, edges_to_csr

__version__ = "0.1.0"
__all__ = ["MultilayerNetwork", "edges_to_csr", "__version__"]
