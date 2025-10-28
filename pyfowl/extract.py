import numpy as np
import configparser
from pyfowl.twopoint import TwoPointFile

# FITS header keyword indicating 2-point data
DEFAULT_PLACEHOLDER = -9999


class TwoPointExtraction():
    # This is an edited verison of TwoPointLikelihood.
    # Instead of computing the actual likelihood, it'll just do the data & cov extraction

    def __init__(self, file_data, file_cuts):
        self.like_name = "2pt" 
        self.placeholder = DEFAULT_PLACEHOLDER
        self.datafile = file_data
        self.config = configparser.ConfigParser()
        self.config.read(file_cuts)

        # self.data_y,self.bin1,self.bin2,self.angle,self.angle_min,self.angle_max, self.arraysizes=
        self.build_data()
        self.cov = self.build_covariance()  # should have cuts applied during build_data
#         self.inv_cov = self.build_inverse_covariance()
#         self.log_det = self.extract_covariance_log_determinant()
#         self.norm = -0.5 * self.log_det

        # get info about what theory points we want to be placeholders
        # in same format as data_sets option; list of 2pt function names
#         placeholders_for = self.options.get_string("placeholders_for", default="none")
#         if placeholders_for != "none":
#             self.placeholders_for = placeholders_for.split()
#         else:
#             self.placeholders_for = []
#         if len(self.placeholders_for):
#             print("In TwoPointExtraction, returning placeholder value {0} for all theory calculations for:".format(
#                 self.placeholder), self.placeholders_for)

    def build_data(self):
        print("In build_data")
        
        # This is the main work - read data in from the file
        self.two_point_data = TwoPointFile.from_fits(self.datafile, covmat_name="COVMAT")

        # Potentially cut out lines. For some reason one version of
        # this file used zeros to mark masked values.
        # if self.options.get_bool("cut_zeros", default=False):
        #     print("Removing 2-point values with value=0.0")
        #     self.two_point_data.mask_bad(0.0)

        # if self.options.get_bool("cut_cross", default=False):
        #     print("Removing 2-point values from cross-bins")
        #     self.two_point_data.mask_cross()

        # All the names of two-points measurements that were found in the data file
        all_names = [spectrum.name for spectrum in self.two_point_data.spectra]

        # We may not want to use all the likelihoods in the file. We can set an option to only use some of them
#         data_sets = self.options.get_string("data_sets", default="all")
#         if data_sets != "all":
#             data_sets = data_sets.split()
#             self.two_point_data.choose_data_sets(data_sets)

        # The ones we actually used.
        self.used_names = [spectrum.name for spectrum in self.two_point_data.spectra]

        # Check for scale cuts. In general, this is a minimum and maximum angle for
        # each spectrum, for each redshift bin combination. Which is clearly a massive pain... but what can you do?
        scale_cuts = {}
        for name in self.used_names:
            s = self.two_point_data.get_spectrum(name)
            for b1, b2 in s.bin_pairs:
                option_name = "angle_range_{}_{}_{}".format(name, b1, b2)
                #TODO Need to read the config file here!
                if option_name in self.config['cuts'].keys():
                    r = self.config['cuts'][option_name]
                    r = [float(n) for n in r.split()]
                    scale_cuts[(name, b1, b2)] = r

        # Now check for completely cut bins
        # example:
        # cut_wtheta = 1,2  1,3  2,3
        #TODO Do I need this?
        bin_cuts = []
        for name in self.used_names:
            s = self.two_point_data.get_spectrum(name)
            option_name = "cut_{}".format(name)
            if option_name in self.config['cuts'].keys():
                cuts = self.config['cuts'][option_name].split()
                cuts = [eval(cut) for cut in cuts]
                for b1, b2 in cuts:
                    bin_cuts.append((name, b1, b2))

        if scale_cuts or bin_cuts:
            #GDA Here we get the mask from the twopoint, they are self.two_point_data.masks
            # Can also return from the function
            self.two_point_data.mask_scales(scale_cuts, bin_cuts)
        else:
            print("No scale cuts mentioned in ini file.")

        # Info on which likelihoods we do and do not use
        print("Found these data sets in the file:")
        total_data_points = 0
        final_names = [
            spectrum.name for spectrum in self.two_point_data.spectra]
        for name in all_names:
            if name in final_names:
                data_points = len(self.two_point_data.get_spectrum(name))
            else:
                data_points = 0
            if name in self.used_names:
                print("    - {}  {} data points after cuts {}".format(name, data_points, "  [using in likelihood]"))
                total_data_points += data_points
            else:
                print("    - {}  {} data points after cuts {}".format(name,
                                                                      data_points, "  [not using in likelihood]"))
        print("Total data points used = {}".format(total_data_points))

        # Convert all units to radians.  The units in cosmosis are all
        # in radians, so this is the easiest way to compare them.
        #GDA This may conflict with our own conversion
        for spectrum in self.two_point_data.spectra:
            if spectrum.is_real_space():
                spectrum.convert_angular_units("rad")

        # build up the data vector from all the separate vectors, just concatenation
        data_vector = np.concatenate(
            [spectrum.value for spectrum in self.two_point_data.spectra])

        # Make sure it's not empty, and keep track of indices
        if len(data_vector) == 0:
            raise ValueError(
                "No data was chosen to be used from 2-point data file {0}. It was either not selectedin data_sets or cut out".format(filename))
        else:
            self.data_y = data_vector
            # keep track of indices where different 2pt functions start
            dataset_names = [spectrum.name for spectrum in self.two_point_data.spectra]
            arraysizes = [spectrum.bin1.size for spectrum in self.two_point_data.spectra]
            self.indexdict = {}
            startind = 0
            for i, name in enumerate(dataset_names):
                self.indexdict['_'.join([self.like_name, name, "startind"])] = startind
                startind += arraysizes[i]
                self.indexdict['_'.join([self.like_name, name, "endind"])] = startind - 1

            self.bin1 = np.concatenate(
                [spectrum.bin1 for spectrum in self.two_point_data.spectra])
            self.bin2 = np.concatenate(
                [spectrum.bin2 for spectrum in self.two_point_data.spectra])
            self.angle = np.concatenate(
                [spectrum.angle for spectrum in self.two_point_data.spectra])
            if self.two_point_data.spectra[0].angle_min is not None:
                self.angle_min = np.concatenate(
                    [spectrum.angle_min for spectrum in self.two_point_data.spectra])
                self.angle_max = np.concatenate(
                    [spectrum.angle_max for spectrum in self.two_point_data.spectra])
            else:
                self.angle_min = None
                self.angle_max = None

        # The x data is not especially useful here, so return None.
        # We will access the self.two_point_data directly later to determine ell/theta values
        return  # data_vector,bin1,bin2,angle,angle_min,angle_max,arraysizes

    def build_covariance(self):
        # covariance
        C = np.array(self.two_point_data.covmat)
        return C

    def extract_theory_points(self):
        #TODO This needs to be adapted
        theory = []
        # We have a collection of data vectors, one for each spectrum
        # that we include. We concatenate them all into one long vector,
        # so we do the same for our theory data so that they match

        # We will also save angles and bin indices for plotting convenience,
        # although these are not actually used in the likelihood
        th_angle = []
        th_bin1 = []
        th_bin2 = []

        # Now we actually loop through our data sets
        for spectrum in self.two_point_data.spectra:
            #  ADDING FOR EXTRACTION
            if spectrum.name in self.placeholders_for:
                #print(spectrum.name, "IS IN PLACEHOLDERS")
                # We want this to work even if the theory calculations haven't been done
                # for these data vectors components
                # just get angle and bin values, put placeholder value in theory vector spots
                theory_vector, angle_vector, bin1_vector, bin2_vector = self.extract_spectrum_placeholder(
                    block, spectrum)
            else:
                theory_vector, angle_vector, bin1_vector, bin2_vector = self.extract_spectrum_prediction(
                    block, spectrum)
            theory.append(theory_vector)
            th_angle.append(angle_vector)
            th_bin1.append(bin1_vector)
            th_bin2.append(bin2_vector)

        # We also collect the ell or theta values.
        # The gaussian likelihood code itself is not expecting these,
        # so we just save them here for convenience.

        th_angle = np.concatenate(th_angle)
        if th_angle.size != self.angle.size:
            print("ANGLE ARRAY SIZES DON'T MATCH between data and theory")
        elif not np.all(th_angle == self.angle):
            print("ANGLE ARRAYS DON'T MATCH between data and theory")
        th_bin1 = np.concatenate(th_bin1)
        th_bin2 = np.concatenate(th_bin2)
        #block.put_double_array_1d(names.data_vector, self.like_name + "_th_bin1", th_bin1)
        #block.put_double_array_1d(names.data_vector, self.like_name + "_th_bin2", th_bin2)
        #block.put_double_array_1d(names.data_vector, self.like_name + "_th_angle", th_angle)
        if (not np.all(th_bin1 == self.bin1)) or (not np.all(th_bin2 == self.bin2)):
            raise ValueError("BIN ARRAYS DO NOT MATCH between data and theory")

        # this will be the theory vector in same order as data vector
        theory = np.concatenate(theory)
        return theory

    def extract_spectrum_prediction(self, spectrum):
        #TODO This needs to be adapted!
        # We may need theory predictions for multiple different
        # types of spectra: e.g. shear-shear, pos-pos, shear-pos.
        # So first we find out from the spectrum where in the data
        # block we expect to find these - mapping spectrum types
        # to block names
        section, x_name, y_name = theory_names(spectrum)

        # To handle multiple different data sets we allow a suffix
        # to be applied to the section names, so that we can look up
        # e.g. "shear_cl_des" instead of just "shear_cl".
        section += self.suffix

        # We need the angle (ell or theta depending on the spectrum)
        # for the theory spline points - we will be interpolating
        # between these to get the data points
        angle_theory = block[section, x_name]

        # Now loop through the data points that we have.
        # For each one we have a pairs of bins and an angular value.
        # This assumes that we can take a single sample point from
        # each theory vector rather than integrating with a window function
        # over the theory to get the data prediction - this will need updating soon.
        bin_data = {}
        theory_vector = []

        # For convenience we will also return the bin and angle (ell or theta)
        # vectors for this bin too.
        angle_vector = []
        bin1_vector = []
        bin2_vector = []
        for (b1, b2, angle) in zip(spectrum.bin1, spectrum.bin2, spectrum.angle):
            # We are going to be making splines for each pair of values that we need.
            # We make splines of these and cache them so we don't re-make them for every
            # different theta/ell data point
            if (b1, b2) in bin_data:
                # either use the cached spline
                theory_spline = bin_data[(b1, b2)]
            else:
                # or make a new cache value
                # load from the data block and make a spline
                # and save it
                if block.has_value(section, y_name.format(b1, b2)):
                    theory = block[section, y_name.format(b1, b2)]
                # It is okay to swap if the spectrum types are the same - symmetrical
                elif block.has_value(section, y_name.format(b2, b1)) and spectrum.type1 == spectrum.type2:
                    theory = block[section, y_name.format(b2, b1)]
                else:
                    raise ValueError("Could not find theory prediction {} in section {}".format(
                        y_name.format(b1, b2), section))
                #theory_spline = interp1d(angle_theory, theory)
                theory_spline = SpectrumInterp(angle_theory, theory)
                bin_data[(b1, b2)] = theory_spline
                # This is a bit silly, and is a hack because the
                # book-keeping is very hard.
                bin_data[y_name.format(b1, b2)] = theory_spline

            # use our spline - interpolate to this ell or theta value
            # and add to our list
            try:
                theory = theory_spline(angle)
            except ValueError:
                raise ValueError("""Tried to get theory prediction for {} {}, but ell or theta value ({}) was out of range.
                    "Maybe increase the range when computing/projecting or check units?""".format(section, y_name.format(b1, b2), angle))
            theory_vector.append(theory)
            angle_vector.append(angle)
            bin1_vector.append(b1)
            bin2_vector.append(b2)

        # Return the whole collection as an array
        theory_vector = np.array(theory_vector)

        # For convenience we also save the angle vector (ell or theta)
        # and bin indices
        angle_vector = np.array(angle_vector)
        bin1_vector = np.array(bin1_vector, dtype=int)
        bin2_vector = np.array(bin2_vector, dtype=int)

        return theory_vector, angle_vector, bin1_vector, bin2_vector
