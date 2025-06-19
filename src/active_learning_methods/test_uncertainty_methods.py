import torch
from uncertainty import calculate_uncertainty
import numpy as np

test_cases = [
{'inputs':
    {'select_uncertainty':'Least_confidence_sampling', 
     'outputs':torch.tensor([[3.0, 2.0, 4.0, 1.0]]),
     'number_of_classes':4},
'expected':np.array([0.4748])},
{'inputs':
    {'select_uncertainty':'Margin_sampling', 
     'outputs':torch.tensor([[3.0, 2.0, 4.0, 1.0]]),
     'number_of_classes':4},
'expected':np.array([0.5930])},
{'inputs':
    {'select_uncertainty':'Entropy_sampling', 
     'outputs':torch.tensor([[3.0, 2.0, 4.0, 1.0]]),
     'number_of_classes':4},
'expected':np.array([0.684])},
{'inputs':
    {'select_uncertainty':'none', 
     'outputs':torch.tensor([[3.0, 2.0, 4.0, 1.0]]),
     'number_of_classes':4},
'expected':np.array([0.0])},

{'inputs':
    {'select_uncertainty':'Least_confidence_sampling',
     'outputs':torch.tensor([[3.0, 3.0, 3.0]]),
     'number_of_classes':3},
'expected':np.array([1])},
{'inputs':
    {'select_uncertainty':'Margin_sampling', 
     'outputs':torch.tensor([[3.0, 3.0, 3.0]]),
     'number_of_classes':3},
'expected':np.array([1])},
{'inputs':
    {'select_uncertainty':'Entropy_sampling', 
     'outputs':torch.tensor([[3.0, 3.0, 3.0]]),
     'number_of_classes':3},
'expected':np.array([1])},
{'inputs':
    {'select_uncertainty':'none', 
     'outputs':torch.tensor([[3.0, 3.0, 3.0]]),
     'number_of_classes':3},
'expected':np.array([0.0])},
]


def test_calculate_uncertainty():
    for test_case in test_cases:
        inputs = test_case['inputs']
        expected = test_case['expected']
        if isinstance(expected, ValueError):
            try:
                calculate_uncertainty(**inputs)
                assert False, f"Expected a ValueError for inputs {inputs}"
                
            except ValueError:
                pass
        else:
            try:
                result = calculate_uncertainty(**inputs)
                assert np.allclose(result, expected, atol=0.001), f"Expected {expected}, but got {result}"
                # print in green color that test passed
                print("\033[92mTest passed\033[0m")
                
            except AssertionError:
                # print in red color that test failed
                print("\033[91mTest failed\033[0m")
                print(f"inputs: {inputs}")
                print(f"expected: {expected}")
                print(f"got: {result}")

test_calculate_uncertainty()