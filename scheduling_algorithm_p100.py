import pandas as pd
import numpy as np
import pyperclip
import io


class PlanningModesP100:
    first_part_two_shifts = 'first_part_two_shifts'
    first_part_one_shift = 'first_part_one_shift'
    second_part_two_shifts = 'second_part_two_shifts'
    second_part_one_shift = 'second_part_one_shift'
    third_part_two_shifts = 'third_part_two_shifts'
    third_part_one_shift = 'third_part_one_shift'
    third_part_740 = 'third_part_740'


class ProductionOrderSchedulerP100:
    SMALL_ORDER_THRESHOLD = 3  # Threshold for small orders in quantity
    SMALL_ORDERS_MAX_SEQUENCE = 3  # Maximum sequence of small orders in the production plan
    TRIPLE_GLAZED_PANES = ['9', '9C']
    URGENT_ORDERS_RECEIVERS_2_pm = ['2101/Polska/C', '3301/Węgry/C']
    SAP_NUMBERS_FOR_FIRST_AND_LAST_POSITIONS = ['808965', '808966', '839134', '839135']
    TWO_SHIFTS_THRESHOLD = 180  # Threshold for two shifts in quantity
    MIDDLE_POINT_PROPORTION = 0.55  # Proportion of the sum of windows per shift to determine the middle point
    ADDITIONAL_MILLING_WIDTHS = [1340]  # Widths that require additional milling operations
    ADDITIONAL_MILLING_VARIANTS = ['EXL', 'PRO']  # Variants that require additional milling operations
    MILLED_WINDOWS_MAX_SEQUENCE = 14  # Maximum sequence of milled windows in the production plan [pcs]
    MILLED_WINDOWS_MIN_SEQUENCE_SEPARATION = 6  # Separation between milled windows in the production plan [pcs]
    MILLED_WINDOWS_TOLERANCE = 2  # Tolerance for milled windows sequence in the production plan [pcs]
    MINIMUM_GAP_BETWEEN_COLORS = 6  # Allowed gap between colors in the production plan [pcs]
    NOT_SCHEDULED_NUMBER = 9999  # number used in Scheduling position column, it shows posistions which were not scheduled

    def __init__(self):
        """
        Initialize the scheduler with production data
        """
        self.production_plan_df = None

        self.current_record_num = 1

        self.num_of_shifts = None  # One or two shifts?
        self.total_sum_of_windows = 0
        self.middle_point = 0  # Middle point of the production plan - 'kreska', frozen part of the plan

        self.total_num_of_small_orders = 0
        self.total_num_of_first_and_last_positions_orders = 0

        self.sum_of_triple_glazed = 0
        self.sum_of_milled_orders = 0
        self.sum_of_golden_oak_triple = 0
        self.sum_of_pine_triple = 0
        self.sum_of_golden_oak_double = 0
        self.sum_of_pine_double = 0
        self.sum_golden_oak_triple_urgent = 0
        self.sum_pine_triple_urgent = 0
        self.sum_of_r3_triple = 0  # Sum of R39 windows
        self.sum_of_r3_double = 0  # Sum of R39 double glazed windows

        self.unique_widths = list()  # Set to store unique widths of windows
        self.temp_unique_widths = list()  # Temporary set to store unique widths in third part planning
        self.last_width_index = 0  # Index of the last width in the unique widths list

        self.unique_sizes = list()  # Set to store unique sizes of windows
        self.temp_unique_sizes = list()  # Temporary set to store unique sizes in third part planning
        self.last_size_index = 0  # Index of the last size in the unique sizes list

        self.production_order_numbers_for_first_and_last_positions = []  # List to store production order numbers for first and last positions

        self.windows_types = ['R3', 'R4', 'R5', 'R7']
        self.milled_types = ['R4', 'R7']
        self.colors_list = ['K', 'G', 'W']
        self.color_before_middle_point = None  # Color of the windows with triple panes before the middle point

        # quantity of windows in first part of the production plan - triple glazed windows
        self.quantity_of_windows_in_first_part = 0

        # variables defining scheduling process
        self.planning_modes = PlanningModesP100()
        self.sum_of_scheduled_orders = 0
        self.small_orders_sequence = 0  # Counter for small orders sequence
        self.can_be_small_order = True  # Flag to indicate if the order can be small
        self.ignore_small_sequence_condition = False  # Flag to ignore small orders sequence condition
        self.can_be_material_unavailable = False  # Flag to indicate if the order can be material unavailable
        self.force_color = False  # Flag to force color for scheduling
        self.force_type = False  # Flag to force window type for scheduling
        self.possible_colors = None  # Possible colors for scheduling
        self.possible_types = None  # Possible window types for scheduling
        self.quantity_between_colors = 0  # Quantity of windows between colors in the production plan
        self.quantity_scheduled_in_previous_iteration = 0  # Total Quantity of windows scheduled in the previous iteration (total scheduled sum at the end of the loop)
        self.number_of_empty_loops = 0  # Counter for empty loops in the scheduling process
        self.ignore_height_matching_condition = False  # Flag to ignore height matching condition
        self.ignore_width_matching_condition = False  # Flag to ignore width matching condition
        self.skip_widths_until_last_width = False  # Flag to skip widths until the last width in the unique widths list

        # milled windows sequence parameters
        self.milled_windows_sequence = 0
        self.not_milled_windows_sequence = 0
        self.milled_windows_max_sequence = self.MILLED_WINDOWS_MAX_SEQUENCE
        self.milled_windows_min_sequence_separation = self.MILLED_WINDOWS_MIN_SEQUENCE_SEPARATION
        self.can_be_milled = True  # Flag to indicate if the order can be milled
        self.force_milled = False  # Flag to force milled windows for scheduling

        # variables defining last scheduled order
        self.last_order_width = None
        self.last_order_height = None
        self.last_order_size = None
        self.last_order_type = None
        self.last_order_glass = None
        self.last_order_color = None
        self.last_order_is_milled = None
        self.last_order_is_triple = None
        self.last_order_quantity = 0
        self.last_order_is_small = None
        self.last_order_prd_num = None

        # load the data into a DataFrame
        # self.load_production_plan_excel()
        self.load_production_plan_zpp_cserie()

        # add new columns to the DataFrame based on various conditions
        self.is_small_order()
        self.is_triple()
        self.is_urgent_till_2_pm()
        self.is_urgent_till_6_pm()
        self.is_material_available()
        self.add_scheduling_columns()
        self.is_milled_window()
        self.add_size_column()

        self.sort_production_plan()
        self.gather_production_order_numbers_for_first_and_last_positions()

        # Calculate some statistics
        self.calculate_triple_glazed()
        self.calculate_milled()
        self.calculate_golden_oak_and_pine()
        self.calculate_total_sum_of_windows()
        self.calculate_num_of_shifts()  # define num of shifts
        self.count_small_orders()
        self.count_first_and_last_positions_orders()
        self.calculate_quantity_of_windows_in_first_part()
        self.calculate_middle_point()
        self.get_unique_widths()
        self.get_unique_sizes()
        self.count_r39_orders()
        self.count_r39_double_orders()

        self.define_milled_windows_sequence_parameters()
        self.define_color_before_middle_point()

        self.scheduled_orders = []

    @staticmethod
    def ping_pong_iter(lst, start_index=0, steps=10):
        n = len(lst)
        if n == 0 or steps <= 0:
            return

        index = start_index
        direction = 1  # 1 = forward, -1 = backward

        for _ in range(steps):
            yield lst[index]

            # Reverse direction *after* repeating the edge
            if (index == n - 1 and direction == 1) or (index == 0 and direction == -1):
                direction *= -1
            else:
                index += direction

    def display_view(self):
        cols_to_display = [
            'goods_receiver',
            'prd_ord_num',
            'sap_nr',
            'product_name',
            'quantity',
            # 'system_status',
            'glass_type',
            'profile_color',
            'width',
            'height',
            'window_type',
            'variant',
            'is_small',
            'is_triple',
            'is_urgent_till_2_pm',
            'is_urgent_till_6_pm',
            'is_material_available',
            'scheduling_position',
            'is_scheduled',
            'is_milled',
            # 'size',
            # 'copy_pos'
        ]
        self.display_df = self.production_plan_df[cols_to_display]

    def copy_df_index_to_clipboard(self, column_name, new_col_name):
        """
        Copies the specified column from a pandas DataFrame to the clipboard using pyperclip.

        Args:
            column_name (str): The name of the column with scheduling positions.
            new_col_name (str): The name from which data should be copied
        Returns:
            bool: True if the data was copied successfully, False otherwise.
        """

        self.production_plan_df.sort_index(inplace=True)  # Ensure the DataFrame is sorted by index
        self.production_plan_df[column_name] = self.production_plan_df[column_name].apply(
            lambda x: x if x else self.NOT_SCHEDULED_NUMBER)
        self.production_plan_df[new_col_name] = self.production_plan_df[column_name]

        self.production_plan_df.sort_values(by=column_name, inplace=True)  # Sort the DataFrame by scheduling position
        self.mark_middle_point(col_name=new_col_name)
        self.production_plan_df.sort_index(inplace=True)  # Ensure the DataFrame is sorted by index

        # Select the column
        column_data = self.production_plan_df[new_col_name]

        # Convert to string with tab separation
        output = io.StringIO()
        column_data.to_csv(output, sep='\t', header=False, index=False)
        column_string = output.getvalue()
        output.close()

        # Copy the string to the clipboard using pyperclip
        pyperclip.copy(column_string)

        self.production_plan_df.sort_values(by=column_name, inplace=True)  # Sort the DataFrame by scheduling position

    def load_production_plan_excel(self):
        # get data from clipboard
        self.production_plan_df = pd.read_clipboard(sep='\t', header=0, index_col=None, dtype={'sap_nr': str})

    def load_production_plan_zpp_cserie(self):
        zpp_cserie_headers = [
            "record_number",
            "Numer sekwencyjny",
            "Nr zlecenia klienta",
            "Poz. zlec. klienta",
            "goods_receiver",
            "prd_ord_num",
            "sap_nr",
            "product_name",
            "quantity",
            "Data dostępn. mat.",
            "Langtext",
            "system_status",
            "glass_type",
            "profile_color",
            "width",
            "height",
            "Merkmalwert 02",
            "Merkmalwert 17",
            "window_type",
            "Merkmalwert 16",
            "Rozpoczęcie według harmonogramu",
            "Merkmalwert 01",
            "variant",
            "Merkmalwert 03",
            "Rodzaj zlecenia",
            "Merkmalwert 14",
            "Merkmalwert 20",
            "Merkmalwert 12",
            "Kontroler MRP",
            "Merkmalwert 25"
        ]

        headers_order = [
            # 'record_number',
            'goods_receiver',
            'prd_ord_num',
            'sap_nr',
            'product_name',
            'quantity',
            'system_status',
            'glass_type',
            'profile_color',
            'width',
            'height',
            'window_type',
            'variant',
        ]

        header_to_be_deleted = [header for header in zpp_cserie_headers if header not in headers_order]

        self.production_plan_df = pd.read_clipboard(sep='\t', index_col=None, dtype={"sap_nr": str},
                                                    names=zpp_cserie_headers)
        self.production_plan_df.drop(columns=header_to_be_deleted, inplace=True, errors='ignore')
        self.production_plan_df['quantity'] = self.production_plan_df['quantity'].apply(
            lambda x: str(x).replace(',000', '')).astype(int)
        self.production_plan_df = self.production_plan_df[headers_order]  # Reorder the DataFrame columns

    def sort_production_plan(self):
        self.production_plan_df.sort_values(
            by=['is_small', 'is_milled', 'profile_color', 'glass_type', 'width', 'height'],
            ascending=[False, False, True, True, True, True], inplace=True)

    def is_small_order(self):
        """
        Check if the order is small based on its size
        """
        self.production_plan_df['is_small'] = self.production_plan_df['quantity'] < self.SMALL_ORDER_THRESHOLD

    def is_triple(self):
        """
        Check if the order is a triple-glazed window order
        """
        self.production_plan_df['is_triple'] = self.production_plan_df['glass_type'].isin(self.TRIPLE_GLAZED_PANES)

    def calculate_triple_glazed(self):
        """
        Calculate the number of triple-glazed windows in the production plan
        """
        self.sum_of_triple_glazed = self.production_plan_df[self.production_plan_df['is_triple']]['quantity'].sum()

    def calculate_milled(self):
        """
        Calculate the number of EXL and PRO orders in the production plan
        """
        self.sum_of_milled_orders = self.production_plan_df[
            (self.production_plan_df['is_milled']) &
            (~self.production_plan_df['is_triple'])
            ]['quantity'].sum()

    def calculate_golden_oak_and_pine(self):
        """
        Calculate the number of Golden Oak and Pine orders in the production plan
        """
        self.sum_of_golden_oak_triple = self.production_plan_df[
            (self.production_plan_df['profile_color'] == 'G') & (self.production_plan_df['is_triple'])][
            'quantity'].sum()
        self.sum_of_pine_triple = self.production_plan_df[
            (self.production_plan_df['profile_color'] == 'K') & (self.production_plan_df['is_triple'])][
            'quantity'].sum()
        self.sum_of_golden_oak_double = self.production_plan_df[
            (self.production_plan_df['profile_color'] == 'G') & (~self.production_plan_df['is_triple'])][
            'quantity'].sum()
        self.sum_of_pine_double = self.production_plan_df[
            (self.production_plan_df['profile_color'] == 'K') & (~self.production_plan_df['is_triple'])][
            'quantity'].sum()
        self.sum_golden_oak_triple_urgent = self.production_plan_df[(self.production_plan_df['profile_color'] == 'G') &
                                                                    (self.production_plan_df['is_triple']) &
                                                                    (self.production_plan_df[
                                                                        'goods_receiver']).str.endswith('/C')][
            'quantity'].sum()
        self.sum_pine_triple_urgent = self.production_plan_df[(self.production_plan_df['profile_color'] == 'K') &
                                                              (self.production_plan_df['is_triple']) &
                                                              (self.production_plan_df['goods_receiver']).str.endswith(
                                                                  '/C')]['quantity'].sum()

    def calculate_total_sum_of_windows(self):
        """
        Calculate the total sum of windows in the production plan
        """
        self.total_sum_of_windows = self.production_plan_df['quantity'].sum()

    def count_r39_orders(self):
        """
        Count the number of R39 orders with material available in the production plan
        """
        self.sum_of_r3_triple = self.production_plan_df[
            (self.production_plan_df['window_type'] == 'R3') & (self.production_plan_df['is_material_available']) & (
            self.production_plan_df['is_triple'])]['quantity'].sum()

    def count_r39_double_orders(self):
        """
        Count the number of R39 double glazed orders in the production plan
        """
        self.sum_of_r3_double = self.production_plan_df[
            (self.production_plan_df['window_type'] == 'R3') & (~self.production_plan_df['is_triple'])][
            'quantity'].sum()

    def count_small_orders(self):
        """
        Count the number of small orders in the production plan
        """
        self.total_num_of_small_orders = self.production_plan_df[self.production_plan_df['is_small']].shape[0]

    def gather_production_order_numbers_for_first_and_last_positions(self):
        """
        Gather production order numbers for first and last positions in the production plan
        """
        self.production_order_numbers_for_first_and_last_positions = self.production_plan_df[
            (self.production_plan_df['sap_nr'].isin(self.SAP_NUMBERS_FOR_FIRST_AND_LAST_POSITIONS)) &
            (self.production_plan_df['quantity'] >= 12)
            ]['prd_ord_num'].tolist()

    def count_first_and_last_positions_orders(self):
        """
        Count the number of orders that can be either first or last positions in the production plan
        """
        self.total_num_of_first_and_last_positions_orders = len(
            self.production_order_numbers_for_first_and_last_positions)

    def is_urgent_till_2_pm(self):
        """
        Check if the order is urgent and needs to be completed by 2 PM
        """
        self.production_plan_df['is_urgent_till_2_pm'] = self.production_plan_df['goods_receiver'].isin(
            self.URGENT_ORDERS_RECEIVERS_2_pm)

    def is_urgent_till_6_pm(self):
        """
        Check if the order is urgent and needs to be completed by 6 PM
        """
        self.production_plan_df['is_urgent_till_6_pm'] = self.production_plan_df['goods_receiver'].str.endswith('/C',
                                                                                                                na=False) & ~ \
                                                         self.production_plan_df['goods_receiver'].isin(
                                                             self.URGENT_ORDERS_RECEIVERS_2_pm)

    def is_material_available(self):
        """
        Check if the material is available for given production order
        """
        self.production_plan_df['is_material_available'] = self.production_plan_df['system_status'].str.startswith(
            'ZWOL')

    def is_milled_window(self):
        """
        Check if the window has additional milling operations
        """
        self.production_plan_df['is_milled'] = self.production_plan_df.apply(
            lambda row: row['variant'] in self.ADDITIONAL_MILLING_VARIANTS or row[
                'width'] in self.ADDITIONAL_MILLING_WIDTHS, axis=1)

    def add_size_column(self):
        """
        Add a column to the DataFrame with the size of the window in the format 'width x height'
        """
        self.production_plan_df['size'] = self.production_plan_df.apply(lambda row: f"{row['width']} x {row['height']}",
                                                                        axis=1)

    def get_unique_widths(self):
        """
        Get unique widths of windows in the production plan
        """
        self.unique_widths = list(set(self.production_plan_df['width'].unique()))
        self.unique_widths.sort()  # Sort the unique widths for better readability

    def get_unique_sizes(self):
        """
        Get unique sizes of windows in the production plan
        """
        self.unique_sizes = list(set(self.production_plan_df['size'].unique()))
        self.unique_sizes = sorted(self.unique_sizes, key=lambda s: float(s.split(' x ')[0]) * float(s.split(' x ')[1]))

    def calculate_num_of_shifts(self):
        if self.total_sum_of_windows >= self.TWO_SHIFTS_THRESHOLD:
            self.num_of_shifts = 2
        else:
            self.num_of_shifts = 1

    def calculate_middle_point(self):
        """
        Calculate the middle point of the production plan based on the quantity of windows
        Middle point is so called "kreska" which is used to determine the point till which the plan if 'frozen'
        """
        if self.num_of_shifts == 2:
            # two shifts production
            self.middle_point = int(self.MIDDLE_POINT_PROPORTION * (self.total_sum_of_windows // 2))
        else:
            # one shift production
            self.middle_point = int(self.MIDDLE_POINT_PROPORTION * self.total_sum_of_windows)

    def add_scheduling_columns(self):
        """
        Add a column to the DataFrame indicating the scheduling position of each order
        False means that the order is not scheduled yet
        1 means that the order is scheduled for the first position and so on
        """
        self.production_plan_df['scheduling_position'] = None  # Initialize with None
        self.production_plan_df['is_scheduled'] = False  # Initialize with False

    def define_color_before_middle_point(self):
        """
        Define colors of windows with triple panes before the middle point of the production plan
        G or K
        """
        if self.sum_of_golden_oak_triple == 0 and self.sum_of_pine_triple == 0:
            return  # No triple glazed windows to define color
        if self.sum_golden_oak_triple_urgent > self.sum_pine_triple_urgent:
            self.color_before_middle_point = 'G'  # Golden Oak
        elif self.sum_golden_oak_triple_urgent < self.sum_pine_triple_urgent:
            self.color_before_middle_point = 'K'
        else:
            if self.sum_of_golden_oak_triple >= self.sum_of_pine_triple:
                self.color_before_middle_point = 'G'
            else:
                self.color_before_middle_point = 'K'

    def define_milled_windows_sequence_parameters(self):
        """
        Define parameters for milled windows sequence in the production plan
        """
        sum_of_not_milled_for_separation = self.total_sum_of_windows - self.sum_of_triple_glazed - self.sum_of_milled_orders - self.sum_of_r3_double
        num_of_groups = (sum_of_not_milled_for_separation / self.MILLED_WINDOWS_MIN_SEQUENCE_SEPARATION) + 1
        self.milled_windows_max_sequence = int(self.sum_of_milled_orders / num_of_groups)
        self.milled_windows_max_sequence = max(self.milled_windows_max_sequence, self.MILLED_WINDOWS_MAX_SEQUENCE)
        print(f"Sum of not milled: {sum_of_not_milled_for_separation}")
        print(f"num of groups {num_of_groups}")
        print(f"milled windows max seqence {self.milled_windows_max_sequence}")

    def calculate_quantity_of_windows_in_first_part(self):
        """
        Calculate the quantity of windows in the first part of the production plan - triple glazed windows
        """
        if self.sum_of_triple_glazed // 2 < self.middle_point:
            # If the total sum of triple-glazed windows is less than the middle point, set the middle point
            self.quantity_of_windows_in_first_part = self.middle_point
        else:
            # Otherwise, set the quantity to 50% of all triple glazed windows
            self.quantity_of_windows_in_first_part = self.sum_of_triple_glazed // 2

    def mark_middle_point(self, col_name):
        """
        Function marks the middle pointo of the plan (frozen part) by multiplying scheduling position by 100
        """
        cum_sum = 0
        for row in self.production_plan_df.itertuples():
            if cum_sum >= self.middle_point and not row.scheduling_position == self.NOT_SCHEDULED_NUMBER:
                self.production_plan_df.at[row.Index, col_name] *= 100
            cum_sum += row.quantity

    def empty_loop_check(self):
        """
        Check if the loop was empty
        If it was empty, reset the ignore_small_sequence_condition flag
        """
        temp_scheduled_orders = self.sum_of_scheduled_orders - self.quantity_scheduled_in_previous_iteration
        if temp_scheduled_orders == 0:
            self.number_of_empty_loops += 1

        if self.number_of_empty_loops == 2:
            self.ignore_height_matching_condition = True  # Ignore height matching condition after 4 empty loops
            self.skip_widths_until_last_width = True  # Skip widths until the last width in the unique widths list

        if self.number_of_empty_loops == 4:
            self.ignore_width_matching_condition = True  # Ignore width matching condition after 2 empty loops
            self.ignore_height_matching_condition = False

        if self.number_of_empty_loops >= 6:
            self.ignore_width_matching_condition = True  # Ignore width matching condition after 2 empty loops
            self.ignore_height_matching_condition = True

        if self.number_of_empty_loops >= 8:
            self.ignore_small_sequence_condition = True

        if self.number_of_empty_loops >= 10:
            self.force_milled = False  # Stop forcing milled windows after 6 empty loops

        if self.number_of_empty_loops >= 12:
            self.can_be_milled = True  # Allow milled windows after 8 empty loops

        self.quantity_scheduled_in_previous_iteration = self.sum_of_scheduled_orders

    def reset_empty_loops_counter(self):
        """
        Reset the counter for empty loops in the scheduling process and reset the ignore_small_sequence_condition flag
        """
        self.ignore_small_sequence_condition = False  # Reset the flag for ignoring small orders sequence condition
        self.ignore_width_matching_condition = False
        self.ignore_height_matching_condition = False  # Reset the flag for ignoring height matching condition
        self.skip_widths_until_last_width = False  # Reset the flag for skipping widths until the last width in the unique widths list
        self.number_of_empty_loops = 0  # Reset the counter for empty loops

    def handle_small_orders_sequence(self, is_small_order):
        """
        Handle the sequence of small orders in the production plan
        """
        if is_small_order:
            self.small_orders_sequence += 1
            if self.small_orders_sequence >= self.SMALL_ORDERS_MAX_SEQUENCE:
                # If the sequence of small orders reaches the maximum, reset the flag
                self.can_be_small_order = False
        else:
            # If the order is not small, reset the sequence and flag
            self.small_orders_sequence = 0
            self.can_be_small_order = True

    def handle_material_availability(self):
        if self.sum_of_scheduled_orders >= self.middle_point:
            # If the sum of scheduled orders reaches the middle point, allow material unavailability
            self.can_be_material_unavailable = True

    def handle_color_force(self, df_row):
        """
        Handle the color force for scheduling
        If the color is forced, schedule only orders with the same color as the last scheduled order
        """
        if self.last_order_color == "W":
            self.quantity_between_colors += df_row.quantity
            if self.quantity_between_colors >= self.MINIMUM_GAP_BETWEEN_COLORS:
                if self.sum_of_scheduled_orders < self.quantity_of_windows_in_first_part:
                    self.possible_colors = ['W', self.color_before_middle_point]
                else:
                    self.possible_colors = self.colors_list
            else:
                self.possible_colors = ['W']

            return  # If the last order color is white, do not force color
        else:
            self.quantity_between_colors = 0

        if self.can_be_material_unavailable:
            colors_left = self.production_plan_df[
                (~self.production_plan_df['is_scheduled']) &
                (self.production_plan_df['profile_color'] == self.last_order_color) &
                (self.production_plan_df['window_type'].isin(self.possible_types)) &
                (self.production_plan_df['is_triple'] == self.last_order_is_triple)
                ]['quantity'].sum()
        else:
            colors_left = self.production_plan_df[
                (~self.production_plan_df['is_scheduled']) &
                (self.production_plan_df['profile_color'] == self.last_order_color) &
                (self.production_plan_df['window_type'].isin(self.possible_types)) &
                (self.production_plan_df['is_triple'] == self.last_order_is_triple) &
                (self.production_plan_df['is_material_available'])
                ]['quantity'].sum()
        print(f"Colors left for {self.last_order_color}:", colors_left)

        if colors_left > 0:
            # If there are still orders with the same color left, force the color
            self.force_color = True
            self.possible_colors = [self.last_order_color]
        else:
            # If there are no orders with the same color left, reset the flag
            self.force_color = False
            self.possible_colors = ['W']

    def handle_window_type(self, planning_mode):
        """
        Handle the window type for scheduling
        """
        if self.can_be_material_unavailable:
            r3_left = self.production_plan_df[
                (~self.production_plan_df['is_scheduled']) &
                (self.production_plan_df['window_type'] == 'R3') &
                (self.production_plan_df['is_triple'] == self.last_order_is_triple)
                ]['quantity'].sum()
        else:
            r3_left = self.production_plan_df[
                (~self.production_plan_df['is_scheduled']) &
                (self.production_plan_df['window_type'] == 'R3') &
                (self.production_plan_df['is_triple'] == self.last_order_is_triple) &
                (self.production_plan_df['is_material_available'])
                ]['quantity'].sum()

        if planning_mode == self.planning_modes.first_part_two_shifts:
            # If the sum of scheduled orders reaches the quantity of windows in the first part minus R39, force R39 type
            r3_trigger = self.sum_of_scheduled_orders >= self.quantity_of_windows_in_first_part - self.sum_of_r3_triple
        elif planning_mode == self.planning_modes.second_part_two_shifts:
            r3_trigger = self.last_order_type == 'R3' and self.sum_of_r3_double > 0
        elif planning_mode == self.planning_modes.second_part_one_shift:
            # plan r3 when other doubled glazed windows are planned and there are any r3 double
            if self.can_be_material_unavailable:
                windows_left = self.production_plan_df[
                    (~self.production_plan_df['is_scheduled']) &
                    (self.production_plan_df['window_type'] != 'R3') &
                    (~self.production_plan_df['is_triple'])
                    ]['quantity'].sum()
            else:
                windows_left = self.production_plan_df[
                    (~self.production_plan_df['is_scheduled']) &
                    (self.production_plan_df['window_type'] != 'R3') &
                    (~self.production_plan_df['is_triple']) &
                    (self.production_plan_df['is_material_available'])
                    ]['quantity'].sum()
            r3_trigger = self.sum_of_r3_double > 0 and windows_left == 0
        elif planning_mode == self.planning_modes.third_part_one_shift:
            # plan r3 if last order type was R3 and there are any r3 triple in production plan
            r3_trigger = self.last_order_type == 'R3' and self.sum_of_r3_triple > 0
        else:
            r3_trigger = None

        if r3_trigger:
            print(f"R3 left:", r3_left)
            if r3_left > 0:
                self.possible_types = ['R3']
                self.force_type = True
            else:
                self.possible_types = self.windows_types
                self.force_type = False

    def handle_milled_windows_sequence(self, df_row):
        """
        Handle the sequence of milled windows in the production plan
        """
        self.force_milled = False  # Reset the force milled flag at the beginning of the function

        if df_row.is_milled:
            self.milled_windows_sequence += df_row.quantity
            self.not_milled_windows_sequence = 0
        else:
            self.not_milled_windows_sequence += df_row.quantity
            self.milled_windows_sequence = 0

        if self.milled_windows_sequence >= self.milled_windows_max_sequence or (
                self.not_milled_windows_sequence <= self.milled_windows_min_sequence_separation and self.not_milled_windows_sequence > 0):
            # If the sequence of milled windows reaches the maximum or the separation is too small, block further scheduling of milled windows
            self.can_be_milled = False
        else:
            # If the sequence of milled windows is within the limits, allow further scheduling
            self.can_be_milled = True

        # Force milled windows if milled windows sequence is too short
        milled_windows_left = self.production_plan_df[
            (~self.production_plan_df['is_scheduled']) &
            (self.production_plan_df['is_milled'])
            ]['quantity'].sum()

        not_milled_double_left = self.production_plan_df[
            (~self.production_plan_df['is_scheduled']) &
            (~self.production_plan_df['is_triple']) &
            (~self.production_plan_df['is_milled'])
            ]['quantity'].sum()

        if not_milled_double_left == 0:
            self.can_be_milled = True

        if self.possible_types and milled_windows_left > 0:
            self.milled_types = ['R4', 'R7']
            is_milled_type_in_possible_types = any(
                [True if item in self.possible_types else False for item in self.milled_types])
            if self.milled_windows_sequence < self.milled_windows_max_sequence and self.can_be_milled and is_milled_type_in_possible_types:
                self.force_milled = True

    def schedule_one_position(self, df_row, planning_mode):
        """
        Schedule one position in the production plan
        """
        self.production_plan_df.at[df_row.Index, 'scheduling_position'] = self.current_record_num
        self.production_plan_df.at[df_row.Index, 'is_scheduled'] = True
        self.current_record_num += 1

        # Update last scheduled order details
        self.last_order_width = df_row.width
        self.last_order_height = df_row.height
        self.last_order_size = df_row.size
        self.last_order_is_milled = df_row.is_milled
        self.last_order_is_triple = df_row.is_triple
        self.last_order_is_small = df_row.is_small
        self.last_order_type = df_row.window_type
        self.last_order_glass = df_row.glass_type
        self.last_order_color = df_row.profile_color
        self.last_order_quantity = df_row.quantity
        self.last_order_prd_num = df_row.prd_ord_num
        self.last_width_index = self.unique_widths.index(df_row.width)
        self.last_size_index = self.unique_sizes.index(df_row.size)

        self.sum_of_scheduled_orders += df_row.quantity
        print("sum of scheduled orders:", self.sum_of_scheduled_orders)

        self.reset_empty_loops_counter()  # Reset the empty loops counter after scheduling an order

        self.handle_small_orders_sequence(df_row.is_small)
        self.handle_material_availability()
        self.handle_color_force(df_row)
        self.handle_window_type(planning_mode=planning_mode)
        self.handle_milled_windows_sequence(df_row)

    def schedule_first_part_of_production_plan(self, planning_mode):
        """
        Schedule the first part of the producttion plan - Triple glazed windows
        """
        is_planning_finished = False
        self.possible_colors = ['W', self.color_before_middle_point]
        self.possible_types = ['R4', 'R5', 'R7']
        counter = 0

        # Schedule triple glazed windows - first part of the production plan
        for width in self.ping_pong_iter(self.unique_widths, start_index=self.last_width_index,
                                         steps=len(self.unique_widths) * 100):
            if self.skip_widths_until_last_width:
                # If we dropped height condition, we want to come back to last scheduled width so that the width constistency is kept if possible
                if width != self.last_order_width:
                    continue
                else:
                    self.skip_widths_until_last_width = False

            if not self.skip_widths_until_last_width:
                counter += 1

            for row in self.production_plan_df.itertuples():
                if row.is_scheduled:
                    continue
                if row.prd_ord_num in self.production_order_numbers_for_first_and_last_positions:
                    continue
                if not self.ignore_width_matching_condition:
                    # if we are not ignoring width matching condition, we check if the width matches the last scheduled order
                    if self.last_order_width and row.width != self.last_order_width:
                        continue
                if not self.ignore_height_matching_condition:
                    # if we are not ignoring height matching condition, we check if the height matches the last scheduled order
                    if self.last_order_height and row.height != self.last_order_height:
                        continue
                if not self.ignore_small_sequence_condition:
                    if not (self.can_be_small_order == row.is_small) and row.is_small:
                        continue
                if not row.is_material_available and not self.can_be_material_unavailable:
                    continue
                if not row.window_type in self.possible_types:
                    continue
                if row.is_triple and row.profile_color in self.possible_colors and row.width == width:
                    self.schedule_one_position(row, planning_mode=planning_mode)

                if not self.force_color and not self.force_type:
                    # If we are not forcing color or type, check if we reached the quantity of windows in the first part
                    if self.sum_of_scheduled_orders >= self.quantity_of_windows_in_first_part:
                        is_planning_finished = True
                        break

            if divmod(counter, len(self.unique_widths))[1] == 0 and counter != 0:
                # If we have iterated through all unique widths, check if it wasn't empty loop
                # if it was empty loop, we need to switch off small orders sequence
                self.empty_loop_check()

            if is_planning_finished:
                print("First part planning finished.")
                break

    def schedule_second_part_of_production_plan(self, planning_mode):
        """
        Schedule the second part of the production plan - Double glazed windows
        """
        self.quantity_scheduled_in_previous_iteration = 0

        is_planning_finished = False
        is_first_iteration = True

        # self.possible_colors = ['W', self.color_before_middle_point]
        if planning_mode == self.planning_modes.second_part_two_shifts:
            if self.last_order_type == 'R3' and self.sum_of_r3_double > 0:
                # If the last scheduled order was R39 and there are still R39 double glazed windows left, force R3 type
                self.possible_types = ['R3']
                self.force_type = True
                self.force_milled = False  # Reset the force milled flag for the second part
            else:
                self.force_type = False
                self.possible_types = self.windows_types

        if planning_mode == self.planning_modes.second_part_one_shift:
            self.force_type = False
            self.possible_types = self.windows_types
            self.possible_types.remove('R3')

        print("Starting second part of the production plan with possible types:", self.possible_types)
        steps = len(self.unique_widths) * 150
        iterator = self.ping_pong_iter(self.unique_widths, start_index=self.last_width_index, steps=steps)
        counter = 0  # Counter for the number of iterations
        loop_counter = 0  # Counter for the number of loops
        width = next(iterator)

        repeat_iteration_over_df = False  # Flag to indicate if we need to repeat the iteration over the DataFrame

        # Schedule double glazed windows - second part of the production plan
        while counter < steps:
            if self.skip_widths_until_last_width:
                # If we dropped height condition, we want to come back to last scheduled width so that the width constistency is kept if possible
                if width != self.last_order_width:
                    width = next(iterator, None)  # Get the next width from the iterator
                    continue
                else:
                    self.skip_widths_until_last_width = False

            if not is_first_iteration:
                if not repeat_iteration_over_df and not self.skip_widths_until_last_width:
                    counter += 1
                    width = next(iterator, None)  # Get the next width from the iterator
                    if width is None:
                        # If there are no more widths to schedule, break the loop
                        print("No more widths to schedule, breaking the loop.")
                        break
            else:
                is_first_iteration = False

            repeat_iteration_over_df = False  # Reset the flag for each iteration

            for row in self.production_plan_df.itertuples():

                if row.width != width:
                    continue

                # Check if all double glazed windows are scheduled
                double_glazed_left = self.production_plan_df[
                    (~self.production_plan_df['is_scheduled']) &
                    (~self.production_plan_df['is_triple'])
                    ]['quantity'].sum()
                if double_glazed_left == 0:
                    is_planning_finished = True
                    break

                if row.is_scheduled:
                    continue
                if row.is_triple:
                    continue
                if row.prd_ord_num in self.production_order_numbers_for_first_and_last_positions:
                    continue
                if not self.ignore_width_matching_condition:
                    # if we are not ignoring width matching condition, we check if the width matches the last scheduled order
                    if self.last_order_width and row.width != self.last_order_width:
                        continue
                if not self.ignore_height_matching_condition:
                    # if we are not ignoring height matching condition, we check if the height matches the last scheduled order
                    if self.last_order_height and row.height != self.last_order_height:
                        continue
                if not self.ignore_small_sequence_condition:
                    if not (self.can_be_small_order == row.is_small) and row.is_small:
                        continue
                if not row.is_material_available and not self.can_be_material_unavailable:
                    continue
                if not row.window_type in self.possible_types:
                    continue
                if not self.can_be_milled and row.is_milled:
                    continue
                if self.force_milled and not row.is_milled:
                    continue
                if not row.profile_color in self.possible_colors:
                    continue
                if row.is_milled:
                    # Check if the milled windows sequence is within the limits
                    if row.quantity + self.milled_windows_sequence > self.milled_windows_max_sequence + self.MILLED_WINDOWS_TOLERANCE:
                        continue
                if row.width == width:
                    self.schedule_one_position(row, planning_mode=planning_mode)
                    if (not self.last_order_is_milled and not self.last_order_type in [
                        'R3']) or not self.last_order_is_small:
                        repeat_iteration_over_df = True  # Set the flag to repeat the iteration over the DataFrame
                        break

            if divmod(counter, len(self.unique_widths))[1] == 0 and counter != 0 and not repeat_iteration_over_df:
                # If we have iterated through all unique widths, check if it wasn't empty loop
                # if it was empty loop, we need to switch off small orders sequence
                loop_counter += 1
                print(f"Loop counter: {loop_counter}")
                self.empty_loop_check()

            if is_planning_finished:
                print("Second part planning finished.")
                break

    def schedule_third_part_of_production_plan(self, planning_mode):
        self.quantity_scheduled_in_previous_iteration = 0

        is_planning_finished = False
        skip_width_740 = False
        counter = 0

        if planning_mode == self.planning_modes.third_part_two_shifts or planning_mode == self.planning_modes.third_part_one_shift:
            self.force_milled = False
            last_double_width = self.unique_widths[self.last_width_index]
            self.temp_unique_widths = self.unique_widths.copy()

            left_r39_740 = self.production_plan_df[(~self.production_plan_df['is_scheduled']) &
                                                   (self.production_plan_df['window_type'] == 'R3') &
                                                   (self.production_plan_df['width'] == np.int64(740)) &
                                                   (self.production_plan_df['is_triple'])
                                                   ]['quantity'].sum()

            if last_double_width != np.int64(740) and not (left_r39_740 > 0 and self.last_order_type == "R3"):
                self.unique_widths.remove(np.int64(740))  # Remove 740 width if last double glazed window wasn't 740
                self.last_width_index = self.unique_widths.index(
                    last_double_width)  # Update the last width index to the last double glazed window

            left_r39_not_740 = self.production_plan_df[(~self.production_plan_df['is_scheduled']) &
                                                       (self.production_plan_df['window_type'] == 'R3') &
                                                       (self.production_plan_df['width'] != np.int64(740)) &
                                                       (self.production_plan_df['is_triple'])
                                                       ]['quantity'].sum()

            if last_double_width == np.int64(740) and self.last_order_type == 'R3' and left_r39_not_740 > 0:
                skip_width_740 = True

        if planning_mode == self.planning_modes.third_part_740:
            self.unique_widths = self.temp_unique_widths.copy()  # Use the temporary unique widths for the third part

        if planning_mode == self.planning_modes.third_part_one_shift:
            if self.last_order_type == 'R3' and self.sum_of_r3_triple > 0:
                # If the last scheduled order was R39 and there are still R39 triple glazed windows left, force R3 type
                self.possible_types = ['R3']
                self.force_type = True
                self.force_milled = False  # Reset the force milled flag for the third part

        print("Starting third part of the production plan with width index:", self.last_width_index)
        # Schedule triple glazed windows - third part of the production plan
        for width in self.ping_pong_iter(self.unique_widths, start_index=self.last_width_index,
                                         steps=len(self.unique_widths) * 50):
            if self.skip_widths_until_last_width:
                # If we dropped height condition, we want to come back to last scheduled width so that the width constistency is kept if possible
                if width != self.last_order_width:
                    continue
                else:
                    self.skip_widths_until_last_width = False

            if not self.skip_widths_until_last_width:
                counter += 1

            for row in self.production_plan_df.itertuples():
                if skip_width_740 and row.width == 740:
                    left_r39_not_740 = self.production_plan_df[(~self.production_plan_df['is_scheduled']) &
                                                               (self.production_plan_df['window_type'] == 'R3') &
                                                               (self.production_plan_df['width'] != np.int64(740))
                                                               ]['quantity'].sum()
                    if left_r39_not_740 == 0:
                        continue
                if row.is_scheduled:
                    continue
                if row.prd_ord_num in self.production_order_numbers_for_first_and_last_positions:
                    continue
                if not self.ignore_width_matching_condition:
                    # if we are not ignoring width matching condition, we check if the width matches the last scheduled order
                    if self.last_order_width and row.width != self.last_order_width:
                        continue
                if not self.ignore_height_matching_condition:
                    # if we are not ignoring height matching condition, we check if the height matches the last scheduled order
                    if self.last_order_height and row.height != self.last_order_height:
                        continue
                if not self.ignore_small_sequence_condition:
                    if not (self.can_be_small_order == row.is_small) and row.is_small:
                        continue
                if not row.is_material_available and not self.can_be_material_unavailable:
                    continue
                if not row.window_type in self.possible_types:
                    continue
                if row.is_triple and row.profile_color in self.possible_colors and row.width == width:
                    self.schedule_one_position(row, planning_mode=planning_mode)

                if not self.force_color and not self.force_type:
                    # If we are not forcing color or type, check if we reached the quantity of windows in the first part
                    windows_left = self.production_plan_df[
                        (~self.production_plan_df['is_scheduled']) &
                        (~self.production_plan_df['prd_ord_num'].isin(
                            self.production_order_numbers_for_first_and_last_positions))
                        ]['quantity'].sum()
                    if windows_left <= 0:
                        is_planning_finished = True
                        break

            if divmod(counter, len(self.unique_widths))[1] == 0 and counter != 0:
                # If we have iterated through all unique widths, check if it wasn't empty loop
                # if it was empty loop, we need to switch off small orders sequence
                self.empty_loop_check()

            if is_planning_finished:
                print("Third part planning finished.")
                break

    def start_or_finish_the_production_plan(self, planning_mode, num_of_orders_to_plan):
        """Select first or last positions to production plan R79 7/11 and 7/14"""
        self.quantity_scheduled_in_previous_iteration = 0

        counter = 0
        num_of_loops = 10

        for i in range(num_of_loops):
            for row in self.production_plan_df.itertuples():
                if row.is_scheduled:
                    continue
                if not self.ignore_height_matching_condition:
                    # if we are not ignoring height matching condition, we check if the height matches the last scheduled order
                    if self.last_order_height and row.height != self.last_order_height:
                        continue
                if row.prd_ord_num in self.production_order_numbers_for_first_and_last_positions:
                    self.schedule_one_position(row, planning_mode=planning_mode)
                    counter += 1
                    if counter >= num_of_orders_to_plan:
                        return  # Exit the function if we reached the number of orders to plan
            self.empty_loop_check()  # Check if the loop was empty after each iteration

    def main_scheduling_function(self):
        """
        Schedule production orders based on the production plan and defined conditions
        """
        num_of_orders_to_plan_as_first_positions = self.total_num_of_first_and_last_positions_orders // 2

        self.start_or_finish_the_production_plan(planning_mode=self.planning_modes.first_part_two_shifts,
                                                 num_of_orders_to_plan=num_of_orders_to_plan_as_first_positions)

        if self.num_of_shifts == 2:
            self.schedule_first_part_of_production_plan(planning_mode=self.planning_modes.first_part_two_shifts)
        else:
            self.schedule_first_part_of_production_plan(planning_mode=self.planning_modes.first_part_one_shift)

        if self.num_of_shifts == 2:
            self.schedule_second_part_of_production_plan(planning_mode=self.planning_modes.second_part_two_shifts)
        else:
            self.schedule_second_part_of_production_plan(planning_mode=self.planning_modes.second_part_one_shift)

        if self.num_of_shifts == 2:
            self.schedule_third_part_of_production_plan(planning_mode=self.planning_modes.third_part_two_shifts)
        else:
            self.schedule_third_part_of_production_plan(planning_mode=self.planning_modes.third_part_one_shift)

        self.schedule_third_part_of_production_plan(planning_mode=self.planning_modes.third_part_740)
        self.start_or_finish_the_production_plan(planning_mode='third_part',
                                                 num_of_orders_to_plan=self.total_num_of_first_and_last_positions_orders - num_of_orders_to_plan_as_first_positions)

        self.copy_df_index_to_clipboard(column_name='scheduling_position', new_col_name='copy_pos')
        self.display_view()

