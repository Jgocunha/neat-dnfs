#include "neat/field_gene.h"

// field_gene.h includes dynamic-neural-field-composer headers, and a FieldGene builds a
// dynamic-neural-field-composer neural field, so compiling and linking this needs the
// headers and libraries behind neat-dnfs to reach the consumer. Constructing one also
// needs neat-dnfs's experiment configuration, so it only happens when asked to.
int main(int argc, char*[])
{
    if (argc > 1)
    {
        const neat_dnfs::FieldGene gene({ neat_dnfs::FieldGeneType::HIDDEN, 1 });
        return gene.getNeuralField() != nullptr ? 0 : 1;
    }
    return 0;
}
