from django.shortcuts import render, redirect, reverse
from django.views.generic.base import View
from .forms import *
from django.http import HttpResponse
from django.templatetags.static import static
from .filters import *


# Create your views here.
class DashboardView(View):
    def get(self, request, *args, **kwargs):
        context = {}

        return render(request, "landing.html", context)
    
class IntroView(View):
    def get(self, request, *args, **kwargs):
        context = {}
        return render(request, "intro.html", context)
    
class BibView(View):
    def get(self, request, *args, **kwargs):
        context = {}
        return render(request, "bibliography.html", context)
  

# Define a class-based view for handling the highpass filter functionality
class HighView(View):
    # GET request handling
    def get(self, request, *args, **kwargs):
        # Define the image URL
        
        # Prepare the context data with forms and filter results
        context = {
            "highpass_form": HighPassForm(request.POST),
            "butterworth_form": HighButterworthForm(request.POST),
            "freq_img": freq_img(),  # Assuming this function generates a URI for displaying frequency domain image
            "hp_data": ideal_highpass_filter(),  # Assuming this function generates a URI for ideal highpass filter
            "bp_data": butterworth_highpass_filter()  # Assuming this function generates a URI for Butterworth highpass filter
        }
        
        # Render the template with the context data
        return render(request, "high.html", context)
    
    # POST request handling
    def post(self, request, *args, **kwargs):
        # Get form instances from POST data
        hp_form = HighPassForm(request.POST)
        bp_form = HighButterworthForm(request.POST)
        
        # Prepare the context data with form instances
        context = {
            "highpass_form": hp_form,
            "butterworth_form": bp_form
        }
        
        # Check if the form's "order" field is present in the POST data
        if request.POST.get('order'):  # If "order" is present, indicating Butterworth filter
            if bp_form.is_valid():
                # Get cutoff frequency and order values from the form
                context['cutoff_frequency'] = bp_form.cleaned_data['cutoff_frequency']
                context['order'] = bp_form.cleaned_data['order']
                
                # Generate Butterworth highpass filter results and overlay
                context['hp_data'] = ideal_highpass_filter()  # Assuming this function generates a URI for ideal highpass filter
                context['bp_data'], butterworth_hp_filter = butterworth_highpass_filter(
                    cutoff_frequency=context['cutoff_frequency'],
                    order=context['order'],
                    return_filter=True
                )
                context["freq_img"] = freq_img(overlay=butterworth_hp_filter)  # Overlay the filter result on the frequency domain image
        
        else:  # If "order" is not present, indicating ideal highpass filter
            if hp_form.is_valid():
                # Get cutoff frequency value from the form
                context['cutoff_frequency'] = hp_form.cleaned_data['cutoff_frequency']
                
                # Generate ideal highpass filter results and overlay
                context['hp_data'], highpass_filter = ideal_highpass_filter(
                    cutoff=context['cutoff_frequency'],
                    return_filter=True
                )
                context['bp_data'] = butterworth_highpass_filter()  # Assuming this function generates a URI for Butterworth highpass filter
                context["freq_img"] = freq_img(overlay=highpass_filter)  # Overlay the filter result on the frequency domain image
                context['ideal'] = 1  # Indicate that the ideal highpass filter was applied
        
        # Render the template with the updated context data
        return render(request, "high.html", context)


"""
LOW
"""

class LowView(View):
    def get(self, request, *args, **kwargs):
        context = {"lowpass_form":LowPassForm(request.POST),
                   "butterworth_low_form":LowButterworthForm(request.POST),
                   "freq_img":freq_img(),
                   "lp_data": ideal_lowpass_filter(),
                   "bp_low_data": butterworth_lowpass_filter()}
        return render(request, "low.html", context)
    
    def post(self, request, *args, **kwargs):
        lp_form = LowPassForm(request.POST)
        bp_low_form = LowButterworthForm(request.POST)
        context = {"lowpass_form":lp_form,
                   "butterworth_low_form":bp_low_form}
        if request.POST.get('order'): # butterworth
            if bp_low_form.is_valid():
                context['cutoff_frequency'] = bp_low_form.cleaned_data['cutoff_frequency']
                context['order'] = bp_low_form.cleaned_data['order']
                context['lp_data'] = ideal_lowpass_filter()
                context['bp_low_data'], filter = butterworth_lowpass_filter(cutoff_frequency = context['cutoff_frequency'], order = context['order'], return_filter=True)
                context["freq_img"] = freq_img(overlay=filter)
        else: # highpass
            if lp_form.is_valid():
                context['cutoff_frequency'] = lp_form.cleaned_data['cutoff_frequency']
                context['lp_data'], filter = ideal_lowpass_filter(cutoff=context['cutoff_frequency'], return_filter=True)
                context['bp_low_data'] = butterworth_lowpass_filter()
                context["freq_img"] = freq_img(overlay=filter)
                context['ideal'] = 1
        return render(request, "low.html", context)
    
class OtherView(View):
    def get(self, request, *args, **kwargs):
        context = {
                    "bandpass_form":BandpassForm(request.POST),
                    "notch_form":Notch(request.POST),
                    "comb_form":Comb(request.POST),
                    "filter_type":0,
                    "freq_img":freq_img(),
                    "bp_data":ideal_bandpass_filter(),
                    "notch_data":ideal_notch_filter(),
                    "comb_data":ideal_comb_filter()
                    }
        return render(request, "other.html", context)
    def post(self, request, *args, **kwargs):
        bp_form = BandpassForm(request.POST)
        notch_form = Notch(request.POST)
        comb_form = Comb(request.POST)
        context = {
                    "bandpass_form":bp_form,
                    "notch_form":notch_form,
                    "comb_form":comb_form,
                    "filter_type":0,
                    "freq_img":freq_img(),
                    "bp_data":ideal_bandpass_filter(),
                    "notch_data":ideal_notch_filter(),
                    "comb_data":ideal_comb_filter()
                    }
        if request.POST.get('period'):
            if comb_form.is_valid():
                context['period'] = comb_form.cleaned_data['period']
                context['comb_data'], filter = ideal_comb_filter(period= context['period'], return_filter=True)
                context["freq_img"] = freq_img(overlay=filter)
                context["filter_type"] = 2
        elif request.POST.get('radius'):
            if notch_form.is_valid():
                context['radius'] = notch_form.cleaned_data['radius']
                context['notch_data'], filter = ideal_notch_filter(radius= context['radius'], return_filter=True)
                context["freq_img"] = freq_img(overlay=filter)
                context["filter_type"] = 1
        else:
            if bp_form.is_valid():
                context['cutoff_low'] = np.min([bp_form.cleaned_data['cutoff_low'],bp_form.cleaned_data['cutoff_high']])
                context['cutoff_high'] = np.max([bp_form.cleaned_data['cutoff_low'],bp_form.cleaned_data['cutoff_high']])
                context['bp_data'], filter = ideal_bandpass_filter(cutoff_low=context['cutoff_low'], cutoff_high=context['cutoff_high'], return_filter=True)
                context["freq_img"] = freq_img(overlay=filter)

        return render(request, "other.html", context)