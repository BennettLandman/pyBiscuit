| Stage | Subject | Hemi | Result | Rule | Verdict |
|---|---|---|---|---|---|
| input VTK/mask/thickness | 0050952 | lh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050952 | lh | median 0.164 deg, p99 0.671 deg; own spread median 0.303, p99 1.09 | policy 0.01/0.1 deg; Q15 proposed 0.454/1.64 | policy FAIL; Q15 PASS |
| HSD coefficients | 0050952 | lh | max abs 0.0885; own spread 0.237 | policy 1e-4; Q15 0.355 | policy FAIL; Q15 PASS |
| HSD energy E1 (PDF) | 0050952 | lh | image 0.757, ours 0.758 (rel 0.00109); own spread 0.00941 | policy 1e-3; Q15 0.0141 | policy FAIL; Q15 PASS |
| HSD with upstream raw icomesh guess | 0050952 | lh | sphere median 113 deg, E1 1.72 | (documents the Q17 fix) | - |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050952 | lh | surf max abs 7.44e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050952 | lh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050952 | lh | r 0.9997, bias -0.0392 mm, max abs 1.41 mm, exact 0.542 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050952 | lh | r 0.89053, rel p99 0.629, ROI-mean rel 0.45; own spread p99 0.0231 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |
| input VTK/mask/thickness | 0050952 | rh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050952 | rh | median 0.11 deg, p99 0.843 deg; own spread median 0.0488, p99 0.326 | policy 0.01/0.1 deg; Q15 proposed 0.0732/0.489 | policy FAIL; Q15 FAIL |
| HSD coefficients | 0050952 | rh | max abs 0.0538; own spread 0.0183 | policy 1e-4; Q15 0.0274 | policy FAIL; Q15 FAIL |
| HSD energy E1 (PDF) | 0050952 | rh | image 0.74, ours 0.745 (rel 0.00611); own spread 0.00149 | policy 1e-3; Q15 0.00223 | policy FAIL; Q15 FAIL |
| HSD with upstream raw icomesh guess | 0050952 | rh | sphere median 0.223 deg, E1 0.742 | (documents the Q17 fix) | - |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050952 | rh | surf max abs 2.86e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050952 | rh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050952 | rh | r 0.99978, bias -0.0425 mm, max abs 1 mm, exact 0.546 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050952 | rh | r 0.88819, rel p99 0.623, ROI-mean rel 0.459; own spread p99 0.0341 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |
| input VTK/mask/thickness | 0050961 | lh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050961 | lh | median 0.278 deg, p99 1.56 deg; own spread median 0.263, p99 1.64 | policy 0.01/0.1 deg; Q15 proposed 0.395/2.47 | policy FAIL; Q15 PASS |
| HSD coefficients | 0050961 | lh | max abs 0.0407; own spread 0.0394 | policy 1e-4; Q15 0.0592 | policy FAIL; Q15 PASS |
| HSD energy E1 (PDF) | 0050961 | lh | image 0.794, ours 0.792 (rel 0.00246); own spread 0.00692 | policy 1e-3; Q15 0.0104 | policy FAIL; Q15 PASS |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050961 | lh | surf max abs 3.05e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050961 | lh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050961 | lh | r 0.99968, bias -0.0188 mm, max abs 1 mm, exact 0.604 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050961 | lh | r 0.88607, rel p99 0.55, ROI-mean rel 0.391; own spread p99 0.0232 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |
| input VTK/mask/thickness | 0050961 | rh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050961 | rh | median 0.297 deg, p99 2.85 deg; own spread median 0.142, p99 0.93 | policy 0.01/0.1 deg; Q15 proposed 0.213/1.39 | policy FAIL; Q15 FAIL |
| HSD coefficients | 0050961 | rh | max abs 0.0226; own spread 0.0131 | policy 1e-4; Q15 0.0197 | policy FAIL; Q15 FAIL |
| HSD energy E1 (PDF) | 0050961 | rh | image 0.756, ours 0.762 (rel 0.00823); own spread 0.00322 | policy 1e-3; Q15 0.00482 | policy FAIL; Q15 FAIL |
| HSD with upstream raw icomesh guess | 0050961 | rh | sphere median 119 deg, E1 1.62 | (documents the Q17 fix) | - |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050961 | rh | surf max abs 2.01e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050961 | rh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050961 | rh | r 0.99974, bias -0.0265 mm, max abs 1 mm, exact 0.582 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050961 | rh | r 0.89406, rel p99 0.616, ROI-mean rel 0.425; own spread p99 0.0229 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |
| input VTK/mask/thickness | 0050971 | lh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050971 | lh | median 0.104 deg, p99 0.59 deg; own spread median 0.23, p99 1.26 | policy 0.01/0.1 deg; Q15 proposed 0.345/1.88 | policy FAIL; Q15 PASS |
| HSD coefficients | 0050971 | lh | max abs 0.0242; own spread 0.121 | policy 1e-4; Q15 0.181 | policy FAIL; Q15 PASS |
| HSD energy E1 (PDF) | 0050971 | lh | image 0.723, ours 0.725 (rel 0.00294); own spread 0.00221 | policy 1e-3; Q15 0.00331 | policy FAIL; Q15 PASS |
| HSD with upstream raw icomesh guess | 0050971 | lh | sphere median 105 deg, E1 1.96 | (documents the Q17 fix) | - |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050971 | lh | surf max abs 3.43e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050971 | lh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050971 | lh | r 0.99978, bias -0.025 mm, max abs 1 mm, exact 0.547 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050971 | lh | r 0.87568, rel p99 0.613, ROI-mean rel 0.446; own spread p99 0.0271 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |
| input VTK/mask/thickness | 0050971 | rh | all identical: True; mid.area max abs 5e-07 | byte identity | PASS |
| HSD sphere (native, deformed) | 0050971 | rh | median 0.198 deg, p99 0.727 deg; own spread median 0.179, p99 0.78 | policy 0.01/0.1 deg; Q15 proposed 0.268/1.17 | policy FAIL; Q15 PASS |
| HSD coefficients | 0050971 | rh | max abs 0.187; own spread 0.192 | policy 1e-4; Q15 0.288 | policy FAIL; Q15 PASS |
| HSD energy E1 (PDF) | 0050971 | rh | image 0.742, ours 0.75 (rel 0.0118); own spread 0.0107 | policy 1e-3; Q15 0.016 | policy FAIL; Q15 PASS |
| HSD with upstream raw icomesh guess | 0050971 | rh | sphere median 93.4 deg, E1 1.57 | (documents the Q17 fix) | - |
| SphericalRemesh (image coeff) surfaces + stat maps | 0050971 | rh | surf max abs 6.48e-05 mm; maps not E-class: none | E | PASS |
| TRACE | 0050971 | rh | 4/4 files byte-identical; Jaccard sulcal 1, gyral 1 | E (Jaccard >= 0.99) | PASS |
| Sulcal depth | 0050971 | rh | r 0.99978, bias -0.0212 mm, max abs 1 mm, exact 0.54 | E: r >= 0.9999, max/range <= 1e-4 | FAIL |
| LGI (klaplace stand-in) | 0050971 | rh | r 0.90021, rel p99 0.597, ROI-mean rel 0.513; own spread p99 0.027 | Q14 r>=0.9999, p99<=1e-3, ROI<=1e-3 | Q14 FAIL; Q15 FAIL |
