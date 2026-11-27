#include "neat/field_gene.h"

// field_gene.h includes dynamic-neural-field-composer headers, and a FieldGene builds a
// dynamic-neural-field-composer neural field, so compiling and linking this needs the
// headers and libraries behind neat-dnfs to reach the consumer. The function is never
// called (a FieldGene also needs neat-dnfs's experiment configuration), but it has
// external linkage, so it is still emitted and the linker must resolve what it uses.
neat_dnfs::FieldGene makeFieldGene()
{
    return neat_dnfs::FieldGene({ neat_dnfs::FieldGeneType::HIDDEN, 1 });
}

int main()
{
    return 0;
}
