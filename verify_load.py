import logging

from scptensor.datasets import load_dataset

logging.basicConfig(level=logging.INFO)


def test_load():
    try:
        print("Testing loading 'sccope'...")
        container = load_dataset("sccope")
        print(f"✓ Successfully loaded 'sccope': {container}")
        print(f"  Shape: {container.shape}")

        print("\nTesting loading 'plexdia'...")
        container = load_dataset("plexdia")
        print(f"✓ Successfully loaded 'plexdia': {container}")

        print("\nTesting loading 'spatial' (no metadata)...")
        container = load_dataset("spatial")
        print(f"✓ Successfully loaded 'spatial': {container}")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_load()
