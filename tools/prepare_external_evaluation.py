"""Metadata-only command: audit external eligibility without starting any model."""
import argparse
import json
from pathlib import Path

from external_evaluation_data import prepare_external_plan


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--annotations',required=True);parser.add_argument('--images',required=True)
    parser.add_argument('--official-train-images',required=True);parser.add_argument('--official-test-images',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    plan=prepare_external_plan(args.annotations,args.images,args.official_train_images,args.official_test_images)
    Path(args.output).write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(plan['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
