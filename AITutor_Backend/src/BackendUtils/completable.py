class Completable(object):
    """Abstract for the is completed method. Used for Data Structures containing interactable items for determining when to allow a user to interact with future items (current item is completed)."""

    def is_completed(self):
        raise NotImplementedError