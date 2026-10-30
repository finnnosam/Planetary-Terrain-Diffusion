# Four-stage terrain generation with explicit structural fields

This is the proposed direction following the completed 5,000-update categorical/continuous planner comparison. It records the user's four-stage flow. It is a design contract, not an implemented or selected replacement pipeline. No new training is launched by this document.

## Stage responsibilities

| Stage | Input | Output | Responsibility |
|---|---|---|---|
| Land and ocean | Requested land fraction | Land/ocean geometry | Continental shapes and domain geometry at zero sea level |
| General terrain | Land/ocean geometry | Broad signed elevation and bathymetry | Broad uplands, continental interiors and ocean depth envelopes |
| Structural terrain (intermediate) | General terrain plus retained coast/domain information | Structural elevation **and several spatial terrain fields, jointly generated** | Mountain systems, adjacent lower regions, quieter interiors, secondary systems and transitions |
| Detailed terrain | Structural elevation **and structural fields** | Detailed elevation | Ridges, valleys and smaller features consistent with the larger arrangement |

These are four logical generation stages. Multiple channels in the intermediate output do not require separate generators or another learned plan-to-height stage. A joint multi-output intermediate is the proposed default architecture. Any extra renderer would need its own demonstrated benefit and explicit compute accounting.

The land fraction is an input to learned coastline generation. Do not force a requested percentage after generation by moving sea level. Sea level remains zero throughout.

## Minimum structural output contract

Start by defining a small number of meaningful fields rather than adding many correlated output channels:

| Field | Meaning | Distinction it must preserve |
|---|---|---|
| Structural elevation | Physical heights/depths at the intermediate scale | Actual arrangement and gradients before fine texture |
| Mountain-system field | Spatial extent, strength and continuity of regional mountain systems; a continuous corridor/region field | A compact curved or branching system versus a broad ramp or scattered high cells |
| Lower-region field | Connected regional lower terrain, basin floors and inter-range corridors | A connected low region versus isolated negative residuals; lower terrain may be well above sea level |
| Local relief field | Intended local height variation at a declared physical neighbourhood | Quiet plains/plateaus versus active relief, independently of absolute elevation |

The exact channel encoding is still to be established. Width can initially be represented by the spatial extent of a region field; a separate width, direction or hierarchy channel should be added only if the minimal representation cannot express the required organization. Earlier axis channels had little effect through their renderer, so they are not automatically part of this design.

A quiet high plateau must be representable. So must a narrow active range next to a quiet lower region, several secondary systems, and a broad low-relief interior. Mountain and lower-region fields must not reduce to absolute high/low elevation thresholds. A mountain-system field must not be defined solely as "more than 150 m above a local blur": that was already tested and did not make system organization reliable.

These are model outputs, not additional required human controls. Their spatial arrangement is sampled jointly from the available broad context and learned prior. A later user-facing mountain-frequency control would require its own training support; it is not implied here.

## What is materially different from the recent pilots

The earlier joint pilot already generated height plus local category/quiet/direction fields. The later separated planner generated categories or continuous multi-scale height statistics and supplied them to a separately trained renderer. Both experiments helped establish that good supplied structure can be read, but neither reliably generated convincing terrain organization.

Both dedicated planners completed 5,000 updates. Loss and physical consistency improved without a clear system-level terrain-quality gain in the reviewed Alps, California and generated-prefix examples. See the [completed comparison](../outputs/structure/spatial-supervision-v1/autonomous-review-5000/analysis/index.html). A subsequent small [guide-regime probe](../outputs/structure/spatial-supervision-v1/guide-regimes-5000/index.html) shows that clean guides can retain substantially more organization in a rugged training example, while degraded-guide generation is weaker even where the supplied feasible reference remains structured. Four selected diagnostics cannot establish a generalization rate or a unique cause.

The proposed change is therefore not "predict several fields" alone. It is to give the intermediate responsibility for regional **systems**, define training labels that actually express that responsibility, generate elevation and those fields together, and make their use by the detail stage measurable. More heads or a different loss label alone would repeat the previous experiment.

## General terrain must leave room for structural generation

The second stage supplies broad geographic context. It should not make its broad upland footprint an unquestionable final mountain shape. The intermediate must have enough freedom to concentrate relief, create complementary lower terrain and produce realistic transitions within that context.

Review the old constraints explicitly. Sea level/coastal-domain protection is a requirement; an absolute rule that all existing land below 200 m is unchangeable is a different policy. Likewise, rigid regional mean preservation is not the same as preserving broad continental geography. Whether those old policies remain appropriate must be tested in feasible-target construction; neither is silently removed or inherited by this proposal.

The existing supplied-plan successes show that constraints are not the only explanation for failure. Releasing a constraint is not by itself a demonstrated solution. No global elevation shift, land-coverage forcing or relief multiplier is allowed.

## Supervision and coupling

Structural labels will initially have to be derived from DEMs; "semantic" labels do not appear merely because we name a channel. Develop and visually audit region-level definitions using multiple physical neighbourhoods, prominence/contrast, corridor geometry and connected lower terrain. Broad slopes, coastal steps and isolated bumps must not automatically become mountain belts. Quiet elevated terrain and legitimate simple plains must remain valid examples.

For each training example, the height target and all structural fields must describe the **same feasible terrain**. If constraints remove a valley, its label cannot remain as though that valley were still present. Conversely, target construction must not remove the organization the intermediate is supposed to learn. Modern-Earth supervision and smoother PaleoDEM broad inputs must have an explicit compatibility recipe; do not retain the original Earth residual while assuming that teaches all missing broad structure.

Generate height and fields as one coupled sample. The earlier joint result warns that a height head can make weak use of the other predicted channels. Supervision must therefore check agreement between the actual structural elevation and its predicted fields, including their arrangement at multiple physical scales. Consistency is necessary but insufficient: the continuous pilot already learned very consistent smooth terrain.

The detail stage should eventually receive the structural fields alongside height. The selected current meso accepts height conditioning only; it cannot use new channels without a compatible model/training change. Keep that model as a height-only comparison while testing the new contract. A field-conditioned model must be checked by removing/swapping fields at fixed height/noise to establish that its intended behaviour actually depends on them.

## Gates before another substantial run

1. **Label audit.** On a declared mix of mountains, basins, plains, plateaus and transitions, show that the proposed fields distinguish those arrangements. Include the corresponding feasible elevation, not just attractive coloured labels. Use fixed physical scales and regional context; identify padding/edge limitations.
2. **Representation and learning sanity checks.** Establish that the proposed outputs can express organized terrain and that a bounded learning test can learn coherent systems, including training examples and monitored holdouts under clean/degraded guides. Exact reproduction of a particular Earth layout is not required of a stochastic sample; coherent alternatives are acceptable. A small-set fit is a capacity/optimization check, not a claim of generalization.
3. **Joint generation check.** Evaluate multiple seeds of structural elevation and all fields together. Judge compact systems, connected lower regions, secondary ranges, quiet terrain and transitions before adding detailed texture. Reject decorative fields that do not correspond to the elevation.
4. **Detail conditioning check.** Establish that fine generation preserves organization and uses the added fields. Do not credit supplied Earth structure or a stepped Earth guide as autonomous generation.
5. **World evaluation.** Capability examples, area-probability samples, guide strata and declared strongest-feature searches answer different questions. Preserve the existing evaluation framework; do not infer world frequency from these regional diagnostics.

Resolution, model capacity, exact loss terms, field encodings and a replacement data-preparation recipe remain open until these checks resolve their requirements. Keep the four-stage responsibility split fixed during that design work; adding another intermediary is not the default response to failure.

Selected coast/relief/intermediate/meso checkpoints remain unchanged. PlanetC remains refinement of stepped B2W Earth terrain, not a generated round trip.
