"""US Stock Market Trading Calendar

Defines US equity market trading days by specifying exchange holidays. Provides
a custom business day offset (TDay) for date arithmetic that respects market
closures.

Purpose:
    Date calculations in backtesting and data processing must account for actual
    trading days, not just weekdays. This module provides a pandas CustomBusinessDay
    offset that skips weekends and US market holidays.

Usage:
    from mktcalendar import TDay

    # Go back one trading day
    yesterday = today - TDay

    # Go forward 5 trading days
    next_week = today + 5 * TDay

    # Generate trading day range
    dates = pd.date_range(start, end, freq=TDay)

Holidays Included:
    - New Year's Day (January 1)
    - Martin Luther King Jr. Day (3rd Monday in January)
    - Presidents' Day (3rd Monday in February)
    - Good Friday (Friday before Easter)
    - Memorial Day (last Monday in May)
    - Independence Day (July 4)
    - Labor Day (1st Monday in September)
    - Thanksgiving Day (4th Thursday in November)
    - Christmas Day (December 25)

Note:
    Does not include:
    - Early market closes (e.g., day before Thanksgiving)
    - Special closures (e.g., weather, mourning)
    - Half-day trading sessions

    For precise historical trading days, cross-reference with exchange calendars
    or actual price data availability.

Dependencies:
    pandas.tseries.holiday for holiday definitions
    pandas.tseries.offsets for CustomBusinessDay

Used By:
    change_raw.py - Date calculations for SQL queries
    Other date arithmetic throughout salamander module
"""

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay
from pandas.tseries.offsets import CustomBusinessDay


class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25)
    ]


TDay = CustomBusinessDay(calendar=USTradingCalendar())
