from tqdm import tqdm
from abc import ABC, abstractmethod


class Observer(ABC):
    def __init__(self, publisher):
        publisher.register_observer(self)

    @abstractmethod
    def notify(self, publisher, *args, **kwargs):
        pass


class ProgressObserver(Observer):
    def notify(self, publisher, *args, **kwargs):
        progress = kwargs.get("progress")
        if progress:
            with tqdm(total=100) as pbar:
                pbar.bar_format = "Task Completed: {postfix[0]}\n{l_bar}{bar}|"
                pbar.postfix = [kwargs.get('completed_command_description')]
                pbar.update(100 * float(progress))

            return 100 * float(progress)
