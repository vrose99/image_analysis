from django import forms
from django.forms.widgets import NumberInput

class RangeInput(NumberInput):
    input_type = 'range'

class HighPassForm(forms.Form):
    cutoff_frequency = forms.IntegerField(widget=RangeInput, min_value=0, max_value=50, step_size=5, label="Cutoff Frequency:", required=False)

class HighButterworthForm(forms.Form):
    cutoff_frequency = forms.IntegerField(widget=RangeInput, min_value=0, max_value=50, step_size=5,label="Cutoff Frequency:", required=False)
    order = forms.IntegerField(widget=RangeInput, min_value=1, max_value=5, step_size=1, label="Order:",required=False)

class LowPassForm(forms.Form):
    cutoff_frequency = forms.IntegerField(widget=RangeInput, min_value=10, max_value=100, step_size=5, label="Cutoff Frequency:", required=False)

class LowButterworthForm(forms.Form):
    cutoff_frequency = forms.IntegerField(widget=RangeInput, min_value=10, max_value=100, step_size=5,label="Cutoff Frequency:", required=False)
    order = forms.IntegerField(widget=RangeInput, min_value=1, max_value=5, step_size=1, label="Order:",required=False)

class BandpassForm(forms.Form):
    cutoff_low = forms.IntegerField(widget=RangeInput, min_value=10, max_value=100, step_size=5,label="Low Cutoff:", required=False)
    cutoff_high = forms.IntegerField(widget=RangeInput, min_value=20, max_value=120, step_size=5,label="High Cutoff:", required=False)

class Notch(forms.Form):
    radius = forms.IntegerField(widget=RangeInput, min_value=0, max_value=75, step_size=5,label="Notch Radius:", required=False)
   
class Comb(forms.Form):
    period = forms.IntegerField(widget=RangeInput, min_value=2, max_value=20, step_size=1,label="Period:", required=False)