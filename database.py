"""
Face Database module for storing and managing registered face encodings.
Uses pickle serialization for persistent storage.
"""

import os
import pickle


DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
ENCODINGS_FILE = os.path.join(DATASET_DIR, "encodings.pkl")


class FaceDatabase:
    """
    Manages a database of named face encodings persisted as a pickle file.
    """

    def __init__(self):
        """Initialize the database, loading existing data if available."""
        os.makedirs(DATASET_DIR, exist_ok=True)
        self.data = self._load()

    def _load(self):
        """Load encodings from disk."""
        if os.path.isfile(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
                print(f"[INFO] Loaded {len(data['names'])} registered face(s) from database.")
                return data
        return {"names": [], "encodings": []}

    def _save(self):
        """Persist encodings to disk."""
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(self.data, f)

    def register(self, name, encoding):
        """
        Register a new face.

        Args:
            name: Person's name.
            encoding: 128-d numpy array face encoding.
        """
        self.data["names"].append(name)
        self.data["encodings"].append(encoding)
        self._save()
        print(f"[INFO] Registered face for '{name}'. Total: {len(self.data['names'])} face(s).")

    def get_all(self):
        """
        Get all registered names and encodings.

        Returns:
            Tuple of (names_list, encodings_list).
        """
        return self.data["names"], self.data["encodings"]

    def delete(self, name):
        """
        Delete all encodings for a given name.

        Args:
            name: Person's name to remove.

        Returns:
            Number of entries removed.
        """
        indices = [i for i, n in enumerate(self.data["names"]) if n.lower() == name.lower()]

        if not indices:
            print(f"[WARNING] No face found for '{name}'.")
            return 0

        # Remove in reverse order to maintain index integrity
        for idx in sorted(indices, reverse=True):
            self.data["names"].pop(idx)
            self.data["encodings"].pop(idx)

        self._save()
        count = len(indices)
        print(f"[INFO] Deleted {count} encoding(s) for '{name}'.")
        return count

    def list_names(self):
        """
        Get a list of unique registered names.

        Returns:
            Sorted list of unique names.
        """
        return sorted(set(self.data["names"]))

    def count(self):
        """Return total number of registered face encodings."""
        return len(self.data["names"])
