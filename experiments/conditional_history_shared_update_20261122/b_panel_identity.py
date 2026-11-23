import hashlib
import json


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


SPEC = {'fixed_across_epochs_readouts_groups': True,
 'labels_pairs_rowids_saved_once': True,
 'missing_class_auc': None,
 'missing_class_status': 'NA_SINGLE_CLASS',
 'natural_rows_per_train_domain': 512,
 'probe_rows_remain_in_online_training': True,
 'rank_negative_per_cell': 256,
 'rank_pairs_per_cell': 4096,
 'rank_pairs_sampling': 'uniform without replacement from positive_negative '
                        'Cartesian product',
 'rank_positive_per_cell': 256,
 'same_panel_ids_for_clean_gaussian_same_seed': True,
 'train_panel_quota_shortfall': 'engineering data-identity diagnostic; do not '
                                'pad or resample after seeing predictions',
 'train_panels': ['P_A', 'P_B'],
 'train_panels_row_disjoint': True,
 'validation_natural_rows_per_domain': 4096,
 'validation_pair_fallback': 'all available pairs if fewer; missing class => '
                             'NA',
 'validation_pairs_per_cell_for_temporal_stat': 4096,
 'validation_row_cap_rule': 'min(4096, available domain population)',
 'validation_row_fold_seed_independent': True,
 'validation_row_folds': ['V_A', 'V_B']}
HASHES = {'kuairand': {2022: {'P_A': '2e1324d26961171fd163df70976a9379ab076be4ce0ab8671ca1958e12ff54e3',
                     'P_B': '5e3a125c5a7c2dddc6ee69951910cc12b62c5a8e7388a06da0fe047d442cff8e',
                     'V_mech': '62b10d5ef5e3cec24c96d722e2e178231861cc576ce13cb6ba591ad72d564dc2'},
              2023: {'P_A': 'ca9972464b9179a7f3cd8a5a84d2b68e3f513e65e38505828e6dd4f47099c3b4',
                     'P_B': 'e68dbd8a08ec2da12ecbaabbb781526485e6dcbefb6475ee76275917d087731c',
                     'V_mech': '931fb576b064fc5e67a034340f5226cc2e95a30e1e626df648c014fe625428bc'},
              2024: {'P_A': '90081672734935abb64ddf48b2b41c16e3a72929fb7e51717154d526e9ee3413',
                     'P_B': '6b71cb1ffc1003849cf5d513ff3bad69469a3d063c58aa5f1bbef2771b89612c',
                     'V_mech': '344a53717a0b2f5507d7c5201dabe721b0a0f5f2ced7363b6c5085fc9ca92b80'},
              2025: {'P_A': '98f901eeba39569b9ba31d0b786747d8d9e917c5a63f7cc1ab1b1471799884f3',
                     'P_B': 'e54d8a94381a1227abd3ad0f2c24369a9110f15efcda521d72535a713c9963e6',
                     'V_mech': 'b5d4dbb79366de3a6bed36136e7b14a46bf50d490b2fa8cf482f4a9c72bb34ef'},
              2026: {'P_A': 'c897503130df8800d1f423b6d5c9b8fd8bfc0c8557472fac4a92220e052d9885',
                     'P_B': '647cfe550f00abd32a6f322e69d1c6c9d21f80f1d1880eada1d18e1dad8bd061',
                     'V_mech': '46f24d809a381bf5e1f24b96b12d6ba74541a183341022d59f5a92d11c0c3b78'}},
 'movielens': {2022: {'P_A': 'afd341f1bcca2cec13675f1b2e12eca2ab72fcfcaf97b7552968807cea756c2f',
                      'P_B': '89e13300bfd0a6d0ae0f63d432acc215921da67c3d95fecfb7433c2e75f066f9',
                      'V_mech': 'beabdeca89dbb9a2284812738ddc13d8216602cef8377d8a11e9cb7c8b1fa085'},
               2023: {'P_A': 'a83563d900ca96e00b29876631239b8ed0c458f7e669867686ee8f4be64c3091',
                      'P_B': 'eddf7d63cc49953abf4254829385d11a5e757e223c6e6855acebeb433983b584',
                      'V_mech': 'a51e42ba26f59015aeda6ac5e91141b7071d32b45bec735a0c2cadadb40f3606'},
               2024: {'P_A': '791a43c809d8ded82565149260200dcd6a5fbd69328d9c3265638a24c145c580',
                      'P_B': '2fa5165be23cfa24dfe3e4f105ed817f06f5a31e727142c6f0a6926a5b54645b',
                      'V_mech': '874aae397888881099e3b4e3ed57f9ec4314293a14e1344291260cf82d709506'},
               2025: {'P_A': '82b7087ce5b80772e40be7a703fa20e34f4f72e71ef44db790688753b6de41d6',
                      'P_B': '0200664d90b788c35c6ac2d3ab403e09e73cfdd160c3e9d4e782806f580b16d4',
                      'V_mech': 'bacae086b4f8368e3d25054aa8fdd4250b8aded7e7524657e385f1f260c0afb5'},
               2026: {'P_A': '77abdebd2c127719b20845d5dca5139cdf470bb8992792191266ee28feaca88f',
                      'P_B': '80e2b9d23aa8f48500df21367cd8b5254e49904c936da28a1764b9f3fc15bce5',
                      'V_mech': '2aca322c321e1a13339d6bfd353b36fc08a2dbd97ecb1ab019172c229886c6a2'}}}
