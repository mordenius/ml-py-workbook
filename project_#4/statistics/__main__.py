from .. import measures_of_central_tendency as mct
from .. import measures_of_variability as mv
from .. import correlation

test_sample = [1, 2, 2, 3, 4, 4, 5]
test_predicted = [1, 2, 2, 3, 4, 3, 5]

mct.get_quantile(test_sample, 0.25)
mct.get_quantile(test_sample, 0.75)
mct.get_quantile(test_sample, 0.90)

sd = mv.get_standard_deviation(test_sample)
r2 = correlation.get_r_of_squares(test_sample, test_predicted)
adj_r2 = correlation.get_adjusted_r_of_squares(test_sample, test_predicted, 3)
