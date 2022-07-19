class OphysMixin:
    """A mixin class for ophys project data"""
    def __init__(self):
        # If we're in the state of combining behavior and ophys data
        if 'date_of_acquisition_behavior' in self._df and \
                'date_of_acquisition_ophys' in self._df:

            # Prioritize ophys_date_of_acquisition
            self._df['date_of_acquisition'] = \
                self._df['date_of_acquisition_ophys']
            self._df = self._df.drop(
                ['date_of_acquisition_behavior',
                 'date_of_acquisition_ophys'], axis=1)
