# Default mutation rates destroy tuned genomes

With the default `config/neat_dnfs.json` mutation settings, each offspring mutates far more than one gene:

- `mutateFieldGenesPerGenomeProbability` 0.8 x `mutateFieldGenePerGeneProbability` 0.8 means each field gene
  mutates with probability 0.64. Connections behave the same way, through the matching connection keys.
- A share of those mutations are drastic rather than small steps:
  - field kernel-type swaps (gauss <-> mexican hat), at `mutateFieldGeneKernelTypeProbability`;
  - random field-parameter resets, at `mutateFieldGeneNeuralFieldGenerateRandomParametersProbability`;
  - excitation/inhibition flips on connections, at `mutateConnectionGeneConnectionSignalProbability`.

In practice, a population seeded from a working, finely tuned architecture loses it within one generation.
In the sequence-timing seeded control, the population mean fell from 0.87 to about 0.35 after one generation
and stayed flat for 60 generations. Only the unmutated elite kept the seed's fitness.

The `hri-packaging` task already uses gentler settings: genome-level 0.5 and per-gene 0.4, about 20% of genes
per offspring. With those settings, plus the drastic mutations cut to 0.02-0.05, the same seed kept a
population mean of about 0.73 after one generation. It found the missing part of the architecture by
generation 46 (fitness 0.994).

Groups of mutation probabilities must sum to 1, and lowering one member without raising its partner crashes
the run at the first mutation:

- `mutateFieldGeneKernelProbability` + `mutateFieldGeneNeuralFieldProbability` +
  `mutateFieldGeneKernelTypeProbability`
- `mutateFieldGeneNeuralFieldParametersProbability` +
  `mutateFieldGeneNeuralFieldGenerateRandomParametersProbability`
- `mutateConnectionGeneKernelProbability` + `mutateConnectionGeneConnectionSignalProbability` +
  `mutateConnectionGeneKernelTypeProbability`
- `gaussKernelProbability` + `mexicanHatKernelProbability`, in both `FieldGeneConstants` and
  `ConnectionGeneConstants`
