# Description: User defined functions for the iPython notebooks

# import libraries
import pandas as pd
import numpy as np
import pickle as pkl
from scipy.stats import pearsonr
import pingouin as pg
import seaborn as sns

class LysimeterEvents:

    '''
    This class represents a lysimeter where we store info about its percolation
    events.

    Parameters
    ----------
    percolation_hourly_tseries : pd.Series
        A pandas series containing hourly percolation data (mm/h).

    rainfall_hourly_tseries : pd.Series
        A pandas series containing hourly rainfall data (mm/h).

    snowdepth_cm_daily_tseries : pd.Series
        A pandas series containing daily snow depth data. Units MUST be cm.

    Attributes
    ----------
    percolation_hourly_tseries : pd.Series
        A pandas series containing hourly percolation data.

    events_df : pd.DataFrame
        A pandas dataframe containing the events and their characteristics.

    rainfall_events_df : pd.DataFrame
        A pandas dataframe containing the rainfall events and their

    snow_events_df : pd.DataFrame
        A pandas dataframe containing the snow events and their characteristics.


    Methods
    -------
    id_events(
        min_rate_mmh : float = 0.01, forward_window_length_h : int = 24*3,
        min_vol_mm : float = 2, verbose : bool = False
    ):
        Go through the percolation tseries and identify events based on the
        specified criteria. The identified events are stored in the events_df
        dataframe.

    id_rainfall_events(prewindow_h : int = 24 * 4, verbose : bool = False):
        Go through the rainfall tseries and identify rainfall events associated
        with the percolation events. The identified rainfall events are stored
        in the rainfall_events_df dataframe.

    calculate_snow_melt(pre_window_days : int = 5, verbose : bool = False):
        Calculate the snow melt for each event based on the snow depth and
        snow density data.

    checks():
        Check the percolation tseries for any issues.


    '''

    def __init__(
            self, percolation_hourly_tseries : pd.Series,
            rainfall_hourly_tseries : pd.Series,
            snowdepth_cm_daily_tseries : pd.Series,
            soil_sensors_df : pd.DataFrame,
        ):

        self.percolation_hourly_tseries = percolation_hourly_tseries.copy()
        self.rainfall_hourly_tseries = rainfall_hourly_tseries.copy()
        self.snowdepth_cm_daily_tseries = snowdepth_cm_daily_tseries.copy()
        self.soil_sensors_df = soil_sensors_df.copy()


        # init a dataframe to hold the events and their characteristics
        self.events_df = pd.DataFrame(columns=[
            "event_start_dt", "event_end_dt", "event_volume_mm",
            "event_duration_h", "event_max_rate_mmh", "event_avg_rate_mmh",
        ])

        # a placeholder for the rainfall events
        self.rainfall_events_df = pd.DataFrame()

        # a placeholder for the snow events
        self.snow_events_df = pd.DataFrame()

        # a placeholder for the surface soil into
        self.surface_soil_df = pd.DataFrame()

        self.checks()


    def id_events(
            self, min_rate_mmh : float = 0.01, forward_window_length_h : int = 24*3,
            min_vol_mm : float = 2, verbose : bool = False,
            event_count_prefix : str = "Event_"
        ):

        '''
        Go through the percolation tseries and identify events based on the
        specified criteria. The identified events are stored in the events_df
        dataframe.

        Parameters
        ----------
        min_rate_mmh : float, optional
            The minimum percolation rate (mm/h) required for an event to be
            considered valid. The default is 0.01.

        forward_window_length_h : int, optional
            The number of hours to look forward from the current time step to
            determine if the event has ended. The default is 24*3.

        min_vol_mm : float, optional
            The minimum volume (mm) required for an event to be considered
            valid. The default is 2.

        verbose : bool, optional
            If True, print out information about the identified events.

        '''

        # update the events_df
        self.events_df = id_events_(
            events_df=self.events_df, min_vol_mm=min_vol_mm, verbose=verbose,
            percolation_hourly_tseries=self.percolation_hourly_tseries,
            min_rate_mmh=min_rate_mmh,
            forward_window_length_h=forward_window_length_h,
            event_count_prefix=event_count_prefix,
        )

    def id_rainfall_events(self, prewindow_h : int = 24 * 4, verbose : bool = False):

        '''
        Go through the rainfall tseries and identify rainfall events associated
        with the percolation events. The identified rainfall events are stored
        in the rainfall_events_df dataframe.

        Parameters
        ----------
        prewindow_h : int, optional
            The number of hours to look back from the start of the percolation
            event to determine the start of the rainfall event. The default is
            24.

        '''

        self.rainfall_events_df = extract_rainfall_events_(
            events_df=self.events_df,
            rainfall_hourly_tseries=self.rainfall_hourly_tseries,
            prewindow_h=prewindow_h, verbose=verbose,
        )

    def calculate_snow(self, pre_window_days : int = 5, verbose : bool = False):

            '''
            Calculate the snow-related data for each event.

            '''

            self.snow_events_df = calculate_snow_(
                data_snowdepth=self.snowdepth_cm_daily_tseries,
                events_df=self.events_df,
                pre_window_days=pre_window_days,
                verbose=verbose,
            )

    def get_surface_soil_into(self, soil_moist_label, soil_temp_label,
                              hours_around_rainfall=6,
                              meaningful_moisture_increase=0.08,
                              smoothing_window_size=3,
                              soildata_freq_mins=30, verbose=False
    ):

        '''
        Get the surface soil moisture and temperature data for the events.

        Parameters
        ----------
        soil_moist_label : str
            The name of the column containing soil moisture data.

        soil_temp_label : str
            The name of the column containing soil temperature data.

        hours_around_rainfall : int, optional
            The number of hours to look around the rainfall event to determine
            the impact of the rainfall event on the soil moisture. The default
            is 6.

        meaningful_moisture_increase : float, optional
            The minimum increase in soil moisture (%) required to detect a
            meaningful increase in soil moisture. The default is 0.08.

        smoothing_window_size : int, optional
            The size of the window (in number of time steps) to use for
            smoothing the soil moisture data. The default is 3.

        soildata_freq_mins : int, optional
            The frequency of the soil moisture data in minutes. The default is
            30.

        verbose : bool, optional
            If True, print out information.
        '''

        self.surface_soil_df = analyze_surfaceSoil_moisture_temperature_(
            data_soil=self.soil_sensors_df,
            rainfall_events_df=self.rainfall_events_df,
            snow_events_df=self.snow_events_df,
            events_df=self.events_df,
            soil_moist_label=soil_moist_label,
            soil_temp_label=soil_temp_label,
            hours_around_rainfall=hours_around_rainfall,
            meaningful_moisture_increase=meaningful_moisture_increase,
            smoothing_window_size=smoothing_window_size,
            soildata_freq_mins=soildata_freq_mins,
            verbose=verbose,
        )


    def checks(self):

        checks_(
            percolation_hourly_tseries=self.percolation_hourly_tseries,
            rainfall_hourly_tseries=self.rainfall_hourly_tseries,
        )

    def to_pickle(self, filepath):
        """
        Save the LysimeterEvents object to a pickle file.

        Parameters
        ----------
        filepath : str
            The path to the pickle file.
        """
        with open(filepath, "wb") as f:
            pkl.dump(self, f)

    def __repr__(self):
        note = (
            f"LysimeterEvents object\n"
            f"Contains {len(self.events_df)} events\n"
            f"Includes four different dataframes:\n"
            f"  events_df\n"
            f"  rainfall_events_df\n"
            f"  snow_events_df\n"
            f"  surface_soil_df\n"
            "All of the above dataframes share the same index, which is the label"
            " of the percolation event.\n"
        )

        if len(self.events_df) == 0:
            note += (
                "(Note: No events have been identified yet. Please run the id_events()"
                " method to identify events.)\n"
            )
        return note


def id_events_(**kwargs):

    '''
    Identifies events in a percolation time series

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    events_df : pd.DataFrame
        A dataframe containing the events characteristics.

    Notes
    -----
    This will be called by the id_events() method of the LysimeterEvents class.
    Please refer to the docstring of that method for more information.

    '''



    events_df = kwargs.get("events_df").copy()
    percolation_hourly_tseries = kwargs.get("percolation_hourly_tseries").copy()
    min_rate_mmh = kwargs.get("min_rate_mmh")
    forward_window_length_h = kwargs.get("forward_window_length_h")
    min_vol_mm = kwargs.get("min_vol_mm")
    verbose = kwargs.get("verbose")
    event_count_prefix = kwargs.get("event_count_prefix", "Event_")

    # iterate each row
    event_start = False
    event_start_dt = pd.NaT
    event_end_dt = pd.NaT
    event_vol = 0
    event_cntr = 0

    for current_time, current_vol_mm in percolation_hourly_tseries.items():

            if current_vol_mm <= min_rate_mmh or pd.isna(current_vol_mm):
                current_vol_mm = 0

            conditions = (
                bool(current_vol_mm > 0),
                bool(event_start)
            )

            match conditions:

                case (True, False):
                    event_start = True
                    event_start_dt = current_time
                    event_vol += current_vol_mm

                case (True, True):
                    event_vol += current_vol_mm
                    continue

                case (False, True):
                    pass

                case (False, False):
                    continue

                case _:
                    raise ValueError("Invalid combination of conditions")

            # calc volume in the forward window
            forward_volume = percolation_hourly_tseries.loc[
                current_time:current_time + pd.Timedelta(hours=forward_window_length_h)
            ].sum(skipna=True)

            if forward_volume < min_vol_mm:

                if event_vol >= min_vol_mm:
                    event_cntr += 1
                    event_end_dt = current_time - pd.Timedelta(hours=1)
                    event_duration_h = (event_end_dt - event_start_dt).total_seconds() / 3600
                    event_max_rate_mmh = percolation_hourly_tseries.loc[event_start_dt:event_end_dt].max()
                    event_avg_rate_mmh = event_vol / event_duration_h

                    # add the event to the events_df
                    events_df.loc[event_cntr] = [
                        event_start_dt, event_end_dt, event_vol,
                        event_duration_h, event_max_rate_mmh, event_avg_rate_mmh,
                    ]

                    if verbose:
                        print(
                            f"Event {event_cntr} added to events_df\n"
                            f" (start: {event_start_dt}, end: {event_end_dt})"
                            f" (duration: {event_duration_h:.2f} h)\n"
                            f" (volume: {event_vol:.2f} mm)"
                            f" (max rate: {event_max_rate_mmh:.2f} mm/h)"
                            f" (avg rate: {event_avg_rate_mmh:.2f} mm/h)"
                            "\n"
                        )


                event_start = False
                event_start_dt = pd.NaT
                event_end_dt = pd.NaT
                event_vol = 0

    if verbose:
        print(
            F"Found {event_cntr} events in the percolation time series"
        )

    # change the index values to the event_count_prefix_#
    events_df.index = [f"{event_count_prefix}{i}" for i in events_df.index]


    return events_df


def validate_series(series, series_name):
    """
    Validate the properties of a given Pandas Series.

    Parameters
    ----------
    series : pd.Series
        The Pandas Series to validate.
    series_name : str
        The name of the series, used in error messages.

    Raises
    ------
    TypeError
        If the series is not a pd.Series or its index is not pd.DatetimeIndex.
    ValueError
        If the series index is not timezone aware or if the series does not have hourly frequency.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"{series_name} must be a pd.Series")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"{series_name} must have a pd.DatetimeIndex")

    if series.index.tz is None:
        raise ValueError(f"{series_name} must have a timezone aware index")

    if series.index.freq != "h":
        raise ValueError(f"{series_name} must have hourly frequency")

def checks_(**kwargs):
    """
    Checks the inputs of the LysimeterEvents class.

    Parameters
    ----------
    kwargs : dict
        Dictionary containing the series to be checked.

    Returns
    -------
    None

    Notes
    -----
    This will be called by the checks() method of the LysimeterEvents class.
    Please refer to the docstring of that method for more information.
    """
    percolation_hourly_tseries = kwargs.get("percolation_hourly_tseries")
    rainfall_hourly_tseries = kwargs.get("rainfall_hourly_tseries")

    validate_series(percolation_hourly_tseries, "percolation_hourly_tseries")
    validate_series(rainfall_hourly_tseries, "rainfall_hourly_tseries")


def trim_rainfall_series_(rainfall_hourly_series):
    """
    Trims the beginning and end of a rainfall time series to start and end with non-zero rain volume.

    Parameters
    ----------
    rainfall_hourly_series : pd.Series
        A Pandas Series representing hourly rainfall data.

    Returns
    -------
    pd.Series
        The trimmed rainfall time series.

    Notes
    -----
    If the entire series contains only zeros or is empty, an empty series is returned.
    """

    if rainfall_hourly_series.empty:
        return rainfall_hourly_series

    # Find the first non-zero index
    start_index = rainfall_hourly_series.ne(0).idxmax()

    # If the series is all zeros, start_index will point to the first element
    if rainfall_hourly_series[start_index] == 0:
        return pd.Series(dtype=rainfall_hourly_series.dtype)  # Return an empty series of the same dtype

    # Find the last non-zero index
    end_index = rainfall_hourly_series.ne(0).iloc[::-1].idxmax()

    # Slice the series from start_index to end_index
    trimmed_series = rainfall_hourly_series.loc[start_index:end_index]

    return trimmed_series

def calculate_weighted_rainfall_variability_(rain_series):
    """
    Calculates the weighted rainfall variability for a given rainfall series.

    Parameters
    ----------
    rain_series : pd.Series
        A Pandas Series representing rainfall data.

    Returns
    -------
    float
        The weighted rainfall variability.

    Notes
    -----
    If the series is empty or contains only zeros, the function returns NaN.
    """

    if rain_series.empty or rain_series.sum() == 0:
        return float('nan')

    # Calculate weights
    weights = rain_series / rain_series.sum()

    # Calculate the weighted mean rainfall
    mean_rainfall = (rain_series * weights).sum()

    # Calculate the weighted variability
    weighted_variability = ((weights * (rain_series - mean_rainfall) ** 2).sum()) ** 0.5

    return weighted_variability


def extract_rainfall_events_(
        events_df: pd.DataFrame, rainfall_hourly_tseries: pd.Series,
        prewindow_h: int = 24 * 4, postwindow_h: int = 0, verbose: bool = False,
    ):
    """
    Extracts rainfall event characteristics from meteorological data based on specified events.

    Parameters
    ----------
    events_df : pd.DataFrame
        DataFrame containing event start and end datetimes.

    rainfall_hourly_tseries : pd.Series
        Series containing hourly rainfall data.

    prewindow_h : int, optional
        Number of hours to include before the event start time. The default is 24*4.

    postwindow_h : int, optional
        Number of hours to include after the event end time. The default is 0.

    verbose : bool, optional
        If True, print out information about the identified events.

    Returns
    -------
    pd.DataFrame
        DataFrame with characteristics of each rainfall event.

    Notes
    -----
    Assumes that the events_df contains 'event_start_dt' and 'event_end_dt' columns.
    """

    # Create an empty DataFrame for rainfall events
    rainfall_events_df = pd.DataFrame(columns=[
        "rainfall_start_dt", "rainfall_end_dt", "rainfall_volume_mm",
        "rainfall_weighted_variability", "rainfall_peak_intensity_mmh",
        "rainfall_nonzero_duration_h",
        "wet_hours_fraction", # wet hours across !PERCOLATION! event
        "rainfall_average_intensity_mmh", # during non-zero rainfall
        "skewness", "kurtosis", "std", "variance",
    ])

    # Iterate through the events DataFrame
    for rowind, row in events_df.iterrows():

        # Define the time window for the associated rainfall event
        t0_dt = row["event_start_dt"] - pd.Timedelta(hours=prewindow_h)
        t1_dt = row["event_end_dt"] + pd.Timedelta(hours=postwindow_h)
        event_rainfall = rainfall_hourly_tseries.loc[t0_dt:t1_dt]

        if verbose:
            print(
                f"Event {rowind}:\n"
                f"considering rainfall data from {t0_dt} to {t1_dt}\n"
            )

        # Get the rainfall series for the event
        rain_series = trim_rainfall_series_(event_rainfall)

        # Get rainfall characteristics
        if not rain_series.empty:
            rain_t0_dt = rain_series.index[0]
            rain_t1_dt = rain_series.index[-1]

            rain_volume_mm = rain_series.sum()
            weighted_variability = calculate_weighted_rainfall_variability_(rain_series)
            peak_intensity_mmh = rain_series.max()
            nonzero_rain_duration_h = rain_series.ne(0).sum()
            rainfall_average_intensity_mmh = rain_series[rain_series != 0].mean()
            wet_hours_fraction = event_rainfall.ne(0).sum() / len(event_rainfall)
            skewness = rain_series.skew()
            kurtosis = rain_series.kurtosis()
            std = rain_series.std()
            variance = rain_series.var()

            # Add data to the rainfall events DataFrame
            rainfall_events_df.loc[rowind] = [
                rain_t0_dt, rain_t1_dt, rain_volume_mm, weighted_variability,
                peak_intensity_mmh, nonzero_rain_duration_h, wet_hours_fraction,
                rainfall_average_intensity_mmh,
                skewness, kurtosis, std, variance,
            ]

            if verbose:
                print(
                    f"  rainfall start: {rain_t0_dt}\n"
                    f"  rainfall end: {rain_t1_dt}\n"
                    f"  rainfall volume: {rain_volume_mm:.2f} mm\n"
                    f"  rainfall weighted variability: {weighted_variability:.2f}\n"
                    f"  rainfall peak intensity: {peak_intensity_mmh:.2f} mm/h\n"
                )

    return rainfall_events_df


def calculate_snow_(
        data_snowdepth: pd.Series,
        events_df: pd.DataFrame, pre_window_days: int = 4, verbose: bool = False,
    ):

    """
    Calculate snow event related data.

    Parameters
    ----------
    data_snowdepth : pd.Series
        Series containing snow depth data, indexed by date.

    data_snowdensity : pd.Series
        Series containing snow density data, indexed by date.

    events_df : pd.DataFrame
        DataFrame containing percolation events information with start and end dates.

    pre_window_days : int, optional
        Number of days to include in the pre-event window. The default is 5.

    verbose : bool, optional
        If True, print out information about the identified events.

    Returns
    -------
    pd.DataFrame
        DataFrame with calculated snow event data including pre-event depth,
        start depth, end depth, average depth, and melted volume.


    Important
    -----
    The snow depth and density data must be indexed by date.
    The units of the snow depth data must be cm.
    The units of the snow density data must be kg/m3.

    """

    events_df = events_df.copy()

    snow_events_df = pd.DataFrame(columns=[
        "depth_prewindow_cm", "depth_start_cm", "depth_end_cm", "ave_depth_cm",
        "snowcover_present_period_fraction",
        "ave_depth_prewindow_cm",
    ])

    for rowind, row in events_df.iterrows():

        # Calculate the time window for the event
        t0_dt = row["event_start_dt"] - pd.Timedelta(days=pre_window_days)
        t1_dt = row["event_end_dt"].date()

        # Select snow depth data for the event time window
        snowdepth_event = data_snowdepth[t0_dt.date():t1_dt].fillna(0)

        # Calculate depth information
        depth_prewindow_cm = snowdepth_event.loc[t0_dt.date().strftime("%Y-%m-%d")]
        depth_start_cm = snowdepth_event.loc[row["event_start_dt"].date().strftime("%Y-%m-%d")]
        depth_end_cm = snowdepth_event.loc[row["event_end_dt"].date().strftime("%Y-%m-%d")]
        ave_depth_cm = snowdepth_event.mean()
        snowcover_present_period_fraction = (
            snowdepth_event[snowdepth_event > 1].count() / len(snowdepth_event)
        )
        ave_depth_prewindow_cm = snowdepth_event.loc[t0_dt.date().strftime("%Y-%m-%d"):row["event_start_dt"].date().strftime("%Y-%m-%d")].mean()

        if verbose:
            print(
                f"Snow event {rowind}:\n"
                f"  pre-window depth: {depth_prewindow_cm:.2f} cm\n"
                f"  start depth: {depth_start_cm:.2f} cm\n"
                f"  end depth: {depth_end_cm:.2f} cm\n"
                f"  average depth: {ave_depth_cm:.2f} cm\n"
            )

        # Append calculated data to the snow_events_df
        snow_events_df.loc[rowind] = [
            depth_prewindow_cm, depth_start_cm, depth_end_cm, ave_depth_cm,
            snowcover_present_period_fraction,
            ave_depth_prewindow_cm,
        ]

    return snow_events_df

def identify_impact_on_soil_moisture_(
        soil_moisture_series : pd.Series, threshold : float = 0.1,
        window_size : int = 3, time_step_mins : int = 30, verbose : bool = False,
    ):
    """
    Identify the time step at which rain has significantly increased soil moisture.

    Parameters
    ----------
    soil_moisture_series : pd.Series
        Time series of soil moisture data.

    threshold : float, optional
        The minimum absolute increase in soil moisture (%) to be considered significant, by default 0.1.

    window_size : int, optional
        The number of time steps to consider for smoothing and noise reduction, by default 3.

    Returns
    -------
    pd.Timestamp
        The time step immediately before the significant change in soil moisture.

    float
        The soil moisture at the time step immediately before the significant change.

    if no significant change is found, returns NaT and NaN
    """

    original_series = soil_moisture_series.copy()

    # Smooth the series to reduce noise
    smoothed_series = soil_moisture_series.rolling(window=window_size).mean()

    # Calculate the absolute change in soil moisture
    abs_change = smoothed_series.diff()

    # Identify the first instance where the change exceeds the threshold
    significant_change = abs_change[abs_change >= threshold]

    if not significant_change.empty:

        # time step immediately before the significant change
        prior_dt = significant_change.index[0] - pd.Timedelta(minutes=time_step_mins)

        # VWC at the time step immediately before the significant change
        prior_vwc = original_series.loc[prior_dt]

        if verbose:
            print(
                f"Analyzing between {original_series.index[0]} and {original_series.index[-1]}"
                "\n"
                f"Significant change found at {significant_change.index[0]}\n"
                f"Prior time step: {prior_dt}"
                f"Prior VWC: {prior_vwc}\n"
            )

        return prior_dt, prior_vwc

    else:

        if verbose:
            print(
                f"From {original_series.index[0]} to {original_series.index[-1]}\n"
                f"no significant increase in soil moisture was found.\n"
            )

        return pd.NaT, np.NaN


def analyze_surfaceSoil_moisture_temperature_(data_soil, rainfall_events_df, snow_events_df, events_df,
                                          soil_moist_label, soil_temp_label,
                                          hours_around_rainfall=6,
                                          meaningful_moisture_increase=0.08, smoothing_window_size=3,
                                          soildata_freq_mins=30, verbose=True):
    """
    Analyze surface soil moisture and temperature for percolation events, considering rainfall and snowmelt effects.


    Parameters
    ----------
    data_soil : pd.DataFrame
        DataFrame containing soil moisture and temperature data.
    rainfall_events_df : pd.DataFrame
        DataFrame with rainfall event start times.
    snow_events_df : pd.DataFrame
        DataFrame with snow event data including melted volume.
    events_df : pd.DataFrame
        DataFrame with percolation event start times.
    soil_moist_label : str
        Label for soil moisture data in `data_soil`.
    soil_temp_label : str
        Label for soil temperature data in `data_soil`.
    hours_around_rainfall : int, optional
        Number of hours around rainfall event start time to consider, default is 6.
    min_snowmelt : float, optional
        Minimum snowmelt (in mm) to consider snow effect, default is 1.
    meaningful_moisture_increase : float, optional
        Threshold for meaningful moisture increase, default is 0.08 (8%).
    smoothing_window_size : int, optional
        Window size for smoothing in soil moisture analysis, default is 3.
    soildata_freq_mins : int, optional
        Frequency of soil data in minutes, default is 30.
    verbose : bool, optional
        Flag to enable printing of processing details, default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with analyzed soil moisture and temperature data for each event.
    """
    surface_soil_df = pd.DataFrame(columns=[
        "prior_vwc_dt", "prior_vwc", "prior_temp",
        "soiltemp_t0_perc_start", "soiltemp_t0_12h", "soiltemp_t0_24h", "soiltemp_t0_36h",
        "soiltemp_t0_48h", "soiltemp_t0_72h", "soiltemp_t1_perc_end",
        "soiltemp_average_prior4days",
        "vwc_t0_perc_start", "vwc_average_prewindow",

    ])

    events_df = events_df.copy()
    rainfall_events_df = rainfall_events_df.copy()
    snow_events_df = snow_events_df.copy()


    for event in events_df.index:
        # Extract relevant data for the event
        rainfall_event_start = rainfall_events_df.loc[event, "rainfall_start_dt"]
        percolation_event_start = events_df.loc[event, "event_start_dt"]

        # Get surface soil moisture/temp series around the rainfall start dt
        t0 = rainfall_event_start - pd.Timedelta(hours=hours_around_rainfall)
        t1 = rainfall_event_start + pd.Timedelta(hours=hours_around_rainfall)
        event_surfsoil_moist_ts = data_soil.loc[t0:t1, soil_moist_label]
        event_surfsoil_temp_ts = data_soil.loc[t0:t1, soil_temp_label]

        # Extended series up to the start of the percolation event
        event_surfsoil_moist_ts_ext = data_soil.loc[t0:percolation_event_start, soil_moist_label]
        event_surfsoil_temp_ts_ext = data_soil.loc[t0:percolation_event_start, soil_temp_label]

        # get soil temp at the start of the percolation event, 12-24-36-48-72 hours before
        soiltemp_t0 = data_soil.loc[percolation_event_start, soil_temp_label]
        soiltemp_t0_12h = data_soil.loc[percolation_event_start - pd.Timedelta(hours=12), soil_temp_label]
        soiltemp_t0_24h = data_soil.loc[percolation_event_start - pd.Timedelta(hours=24), soil_temp_label]
        soiltemp_t0_36h = data_soil.loc[percolation_event_start - pd.Timedelta(hours=36), soil_temp_label]
        soiltemp_t0_48h = data_soil.loc[percolation_event_start - pd.Timedelta(hours=48), soil_temp_label]
        soiltemp_t0_72h = data_soil.loc[percolation_event_start - pd.Timedelta(hours=72), soil_temp_label]

        soiltemp_average_prior4days = data_soil.loc[
            percolation_event_start - pd.Timedelta(days=4):percolation_event_start, soil_temp_label
        ].mean()

        vwc_average_prewindow = data_soil.loc[
            rainfall_event_start - pd.Timedelta(hours=hours_around_rainfall):rainfall_event_start, soil_moist_label
        ].mean()

        # get soil temp at the end of the percolation event
        soiltemp_t1 = data_soil.loc[events_df.loc[event, "event_end_dt"], soil_temp_label]

        vwc_t0_perc_start = data_soil.loc[percolation_event_start, soil_moist_label]


        if verbose:
            print_processing_details(event, percolation_event_start, rainfall_event_start)

        notes, prior_vwc_dt, prior_vwc, prior_temp = process_no_snow_event(event_surfsoil_moist_ts,
                                                                            event_surfsoil_temp_ts,
                                                                            meaningful_moisture_increase,
                                                                            smoothing_window_size, soildata_freq_mins)


        # Add the data to the dataframe
        # surface_soil_df.loc[event] = [prior_vwc_dt, prior_vwc, prior_temp, notes]
        surface_soil_df.loc[
            event, ["prior_vwc_dt", "prior_vwc", "prior_temp"]
        ] = [prior_vwc_dt, prior_vwc, prior_temp]

        # add other soil temp data
        surface_soil_df.loc[event, [
            "soiltemp_t0_perc_start", "soiltemp_t0_12h", "soiltemp_t0_24h", "soiltemp_t0_36h",
            "soiltemp_t0_48h", "soiltemp_t0_72h", "soiltemp_t1_perc_end",
            "soiltemp_average_prior4days", "vwc_t0_perc_start", "vwc_average_prewindow",
            ]] = [
                soiltemp_t0, soiltemp_t0_12h, soiltemp_t0_24h, soiltemp_t0_36h,
                soiltemp_t0_48h, soiltemp_t0_72h, soiltemp_t1,
                soiltemp_average_prior4days, vwc_t0_perc_start, vwc_average_prewindow,
            ]

    return surface_soil_df

def print_processing_details(event, percolation_event_start, rainfall_event_start):
    """Print details of the processing for each event."""

    print(
        f"Processing event {event}:\n"
        f"Percolation event start: {percolation_event_start}\n"
        f"Rainfall event start: {rainfall_event_start}\n"
    )

def process_no_snow_event(event_surfsoil_moist_ts, event_surfsoil_temp_ts,
                          meaningful_moisture_increase, smoothing_window_size, soildata_freq_mins):
    """Process events with no snow effect."""

    notes = "NoSnowEffect_"

    prior_vwc_dt, prior_vwc = identify_impact_on_soil_moisture_(
        soil_moisture_series=event_surfsoil_moist_ts,
        threshold=meaningful_moisture_increase,
        window_size=smoothing_window_size,
        time_step_mins=soildata_freq_mins, verbose=False,
    )

    if not pd.isna(prior_vwc):
        notes += "FoundPriorVWC"
        prior_temp = event_surfsoil_temp_ts.loc[prior_vwc_dt]
    else:
        notes += "CouldNotFindPriorVWC"
        prior_temp = np.nan

    return notes, prior_vwc_dt, prior_vwc, prior_temp

def process_snow_event(event_surfsoil_moist_ts_ext, event_surfsoil_temp_ts_ext,
                       meaningful_moisture_increase, smoothing_window_size, soildata_freq_mins):
    """Process events with snow effect."""

    notes = "SnowEffect_"

    prior_vwc_dt, prior_vwc = identify_impact_on_soil_moisture_(
        soil_moisture_series=event_surfsoil_moist_ts_ext,
        threshold=meaningful_moisture_increase,
        window_size=smoothing_window_size,
        time_step_mins=soildata_freq_mins, verbose=False,
    )

    if not pd.isna(prior_vwc):
        notes += "FoundPriorVWC"
        prior_temp = event_surfsoil_temp_ts_ext.loc[prior_vwc_dt]
    else:
        notes += "CouldNotFindPriorVWC"
        prior_temp = np.nan

    return notes, prior_vwc_dt, prior_vwc, prior_temp


def load_events(file_path):
    """
    Load a LysimeterEvents object from a pickle file.

    Parameters
    ----------
    file_path : str
        The path to the pickle file.

    Returns
    -------
    LysimeterEvents
        The LysimeterEvents object.
    """

    with open(file_path, "rb") as file:
        return pkl.load(file)

def plot_rectangle_and_text(
        ax, icol, corr, pval, y_pos, color, half_width, fntsize, lwdith=3,
    ):
    '''
    Plot a rectangle with the given correlation coefficient and p-value

    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on

    icol : int
        Column index

    corr : float
        Correlation coefficient

    pval : float
        p-value

    y_pos : float
        y-position of the rectangle

    color : str
        Color of the rectangle

    half_width : float
        Half the width of the rectangle

    fntsize : int
        Font size for the text annotation

    lwdith : int
        Line width of the border.

    Returns
    -------
    None
    '''

    # set dimensions of the rectangle
    add_width = half_width + np.exp(np.abs(corr)) / 7
    total_width = add_width

    # plot rectangle
    ax.fill_between(
        x=[icol - add_width/2, icol + add_width/2],
        y1=[y_pos - total_width/2, y_pos - total_width/2],
        y2=[y_pos + total_width/2, y_pos + total_width/2],
        color=color,
        alpha=0.5,
    )

    # annotate with correlation coefficient
    ax.text(
        icol,
        y_pos,
        f"{corr:.2f}",
        ha="center",
        va="center",
        color="black",
        fontsize=fntsize,
    )

    # draw border around, conditionally
    if pval <= 0.05:
        ax.plot(
            [icol - add_width/2, icol + add_width/2, icol + add_width/2, icol - add_width/2, icol - add_width/2],
            [y_pos - total_width/2, y_pos - total_width/2, y_pos + total_width/2, y_pos + total_width/2, y_pos - total_width/2],
            color="black",
            linewidth=lwdith,
        )


def create_correlation_dataframe(dep_vars):
    '''
    Create a dataframe with the given dependent variables as columns and
    correlation coefficient and p-value as rows.

    Parameters
    ----------
    dep_vars : list
        List of dependent variables

    Returns
    -------
    pd.DataFrame
        Dataframe with correlation coefficients and p-values as rows and
        dependent variables as columns
    '''

    the_df = pd.DataFrame(
        columns=[f"{dep_var}" for dep_var in dep_vars],
        index=["Coeff", "p-value"]
    )

    return the_df

def calculate_pearson_correlation(data, indep_var, dep_var):
    '''
    Calculate the Pearson correlation coefficient and p-value between two
    variables.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data

    indep_var : str
        Name of the independent variable

    dep_var : str
        Name of the dependent variable

    Returns
    -------
    coeff : float
        Pearson correlation coefficient

    '''

    # drop rows with missing values
    data_col = data[[indep_var, dep_var]].dropna()

    # calculate correlation coefficient and p-value
    coeff, pvalue = pearsonr(data_col[indep_var], data_col[dep_var])


    return coeff, pvalue

def calculate_partial_correlation(data, indep_var, dep_var, covars, y_covar=False):
    '''
    Calculate the partial correlation coefficient and p-value between two
    variables, controlling for the given covariates.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data

    indep_var : str
        Name of the independent variable

    dep_var : str
        Name of the dependent variable

    covars : list
        List of covariates

    y_covar : bool
        If True, the effect of the covariates is controlled for in the dependent
        variable. If False, the effect of the covariates is controlled for in
        both the independent and dependent variables.

    Returns
    -------
    coeff : float
        Partial correlation coefficient

    pvalue : float
        p-value
    '''
    # select the columns of interest
    data_col = data.loc[:, [indep_var, dep_var] + covars].copy()

    # calculate correlation coefficient and p-value
    if not y_covar:
        coeff, pvalue = pg.partial_corr(
            data=data_col,
            y=indep_var, x=dep_var,
            covar=covars
        ).loc[:, ["r", "p-val"]].values[0]
    else:
        coeff, pvalue = pg.partial_corr(
            data=data_col,
            y=dep_var, x=indep_var,
            y_covar=covars,
        ).loc[:, ["r", "p-val"]].values[0]


    return coeff, pvalue

def print_event_info(event, event_title=None):
    '''
    Prints the info on the event in a nice format.

    Parameters
    ----------
    event : pd.Series
        The event series.

    event_title : str
        The title of the event.

    Returns
    -------
    None
    '''

    if event_title is None:
        try:
            event_title = event["event_label"]
        except KeyError:
            event_title = "Event ???"

    info_format = {
        "Start": "percolation_start_dt",
        "End": "percolation_end_dt",
        "Volume": ("percolation_volume_mm", "{:.1f} mm"),
        "Duration": ("percolation_duration_h", lambda x: f"{x / 24:.2f} days"),
        "Max intensity": ("percolation_max_intensity_mmh", "{:.2f} mm/h"),
        "Average intensity": ("percolation_ave_intensity_mmh", "{:.2f} mm/h"),
        "Rainfall volume": ("rainfall_volume_mm", "{:.1f} mm"),
        "Simulated input water": ("SVS_water_input_mm", "{:.1f} mm"),
        # "Snowmelt volume": ("snowmelt_volume_mm", "{:.1f} mm"),
        "Antecedent soil moisture": ("surface75mm_antecedent_SoilMoisture_percent", "{:.1f} %"),
        "Antecedent soil temperature": ("surface75mm_antecedent_SoilTemp_celsius", "{:.1f} C"),
        # "Antecedent time": ("surface75mm_antecedent_time_h", "{:.1f} h"),
        # "Soil temperature at event start": ("surface75mm_SoilTemp_PercolationStart_celsius", "{:.1f} C"),
        # "Wet hours fraction": ("wet_hours_fraction", "{:.2f}"),
    }

    # assert if the values are in the dataframe
    # if they are not, do not print them
    not_in_df = []
    for key, value in info_format.items():
        if isinstance(value, tuple):
            attribute, format_spec = value
        else:
            attribute = value

        if attribute not in event:
            not_in_df.append(key)
            print(f"Warning: {attribute} not in the dataframe. {key} will not be printed.")

    print(f"{event_title} or {event.name}")
    for key, value in info_format.items():

        # if the value is not in the dataframe, do not print it
        if key in not_in_df:
            continue

        # if the value is a tuple, it means that it needs to be formatted
        if isinstance(value, tuple):
            attribute, format_spec = value
            formatted_value = format_spec.format(event[attribute]) if isinstance(format_spec, str) else format_spec(event[attribute])

        # if the value is not a tuple, it means that it is a string
        else:
            formatted_value = event[value]
        print(f"{key}: {formatted_value}")

    print("_" * 60)
    print("\n")

def plot_bar(data, ax, x, y, color, edgecolor, linewidth, **kwargs):
    '''
    Plot a bar with the given parameters.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data.

    ax : matplotlib axis
        Axis to plot on.

    x : float
        x-coordinate of the bar.

    y : float
        y-coordinate of the bar.

    color : str
        Color of the bar.

    edgecolor : str
        Edge color of the bar.

    linewidth : float
        Line width of the bar.

    kwargs : dict
        Additional keyword arguments to pass to the barplot function.

    Returns
    -------
    None
    '''

    sns.barplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs
    )
