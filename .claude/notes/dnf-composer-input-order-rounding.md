# Two dnf-composer processes with the same seed do not match bit for bit

The same binary, design and noise seed gives slightly different field activations from one process to the next: up to
1.1e-10 over 12000 steps, on activations of tens to hundreds. A bit-identity test between two library builds therefore
fails even when the two builds are the same.

The likely cause is `Element::inputs`, an `unordered_map<shared_ptr<Element>, string>`. `cachedInputs` follows that
order, so the order inputs are summed in depends on heap addresses, and the rounding depends on the order. This fits the
code and the data but was not confirmed by a dedicated test.

To compare builds, measure the largest activation difference against that noise floor instead of hashing. Run the
reference binary against a copy of itself to get the floor. Round-off of this size has no behavioural effect, but it
does mean two evaluations of one design with one seed cannot be expected to give exactly the same fitness.
