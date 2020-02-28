import argparse

from data_preparation import preprocess_images
from train import train_model
from predict import predict


if __name__ == "__main__":
    print("[LOG] ~GAPING BLOWHOLE~ WHALE IDENTIFICATION")

    print("""
  .-------------'```'----....,,__                        _,
  |                               `'`'`'`'-.,.__        .'(
  |                                             `'--._.'   )
  |                                                   `'-.<
  \               .-'`'-.                            -.    `\\
   \               -.o_.     _                     _,-'`\    |
    ``````''--.._.-=-._    .'  \            _,,--'`      `-._(
      (^^^^^^^^`___    '-. |    \  __,,..--'                 `
       `````````   `'--..___\    |`
                             `-.,' 
""")

    print("[LOG] Parsing command line arguments")
    parser = argparse.ArgumentParser(description="Gaping Blowhol Hackathon Logic")
    parser.add_argument(
        "--dataprep", help="Start data preparation", action="store_true"
    )
    parser.add_argument(
        "--train", help="Start training a whale recognision model", action="store_true"
    )
    parser.add_argument(
        "--predict", help="Run model prediction", action="store_true"
    )
    args = parser.parse_args()

    if args.dataprep:
        print("[LOG] Starting data preparation")
        preprocess_images()
    elif args.train:
        print("[LOG] Starting model training")
        train_model()
    elif args.predict:
        print("[LOG] Starting model predict")
        predict()
