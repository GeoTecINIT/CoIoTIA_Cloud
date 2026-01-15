"""
Template to customise the input collection for the models.
You can modify the methods below to change how inputs are collected.
You can add the attributes you need to the InputCollector class.
"""

from abstract.InputCollectorAbstract import InputCollectorAbstract

class InputCollector(InputCollectorAbstract):
    def __init__(self):
        super().__init__()
        # Add any additional attributes you need here
        pass

    def collect_inputs(self, request):
        # Customize this method to collect inputs as needed
        pass