import os
import sys

sys.path.append("../src")
# __file__-based (not CWD-relative): multiblimp.unimorph needs the repo root
# on sys.path too (for `resources.um2ud_annotation`), and a plain "../src"
# only resolves right if this script happens to be run from scripts/ itself.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_repo_root, "src"))
sys.path.append(_repo_root)

from multiblimp.ud2um import create_all_unimorph_from_ud
from multiblimp.languages import get_ud_langs


if __name__ == "__main__":
    resource_dir = os.path.join(_repo_root, "resources")
    ud_langs = get_ud_langs(resource_dir)

    create_all_unimorph_from_ud(
        ud_langs,
        resource_dir=resource_dir,
        load_from_pickle=True,
        dup_form_threshold=3.0,
        dup_feat_threshold=10.0,
        save_ud2um_stats=True,
    )
