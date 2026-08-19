import argparse
import sys

sys.path.append("../../src/")

from subj_aux.pipeline import SubjAuxPipeline
from multiblimp.languages import get_ud_langs
from multiblimp.swap_features import swap_number_subj_any

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", "-l", nargs="*", help="Languages to process", default=[])
    parser.add_argument("--never_skip", "-ns", action="store_true",
                        help="Always re-extract, re-fit, and re-render everything")
    parser.add_argument("--distinguish_unk", "-d", action="store_true",
                        help="Keep +-/-- labels distinct instead of collapsing them "
                             "into 'unk' before fitting the decision tree (default: collapsed)")
    parser.add_argument("--single_aux_only", action="store_true",
                        help="Use single-aux instances only, excluding stacked-aux "
                             "(2+) ones (default: single+stacked-aux combined)")
    parser.add_argument("--force", action="store_true",
                        help="Skip the startup check for other already-running "
                             "instances of this script. Only use for deliberate "
                             "concurrent runs.")
    args = parser.parse_args()

    resource_dir = "../../resources"

    pipeline = SubjAuxPipeline(
        feature="Number",
        inflection_map=swap_number_subj_any,
        langs=(args.langs if args.langs else get_ud_langs(resource_dir)),
        resource_dir=resource_dir,
        word_order_dir="../../treebank_features/subj_aux",
        never_skip=args.never_skip,
        simplify=not args.distinguish_unk,
        include_multi_aux=not args.single_aux_only,
        force=args.force,
    )
    pipeline()
