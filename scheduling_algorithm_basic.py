import pandas as pd
import pyperclip
import io


class ProductionOrderSchedulerBasic:
    SMALL_ORDER_THRESHOLD = 3  # Threshold for small orders in quantity
    SMALL_ORDERS_MAX_SEQUENCE = 3  # Maximum sequence of small orders in the production plan
    URGENT_ORDERS_RECEIVERS_2_pm = ['2101/Polska/C', '3301/Węgry/C']
    TWO_SHIFTS_THRESHOLD = 180  # Threshold for two shifts in quantity
    MIDDLE_POINT_PROPORTION = 0.6  # Proportion of the sum of windows per shift to determine the middle point
    NOT_SCHEDULED_NUMBER = 9999  # number used in Scheduling position column, it shows posistions which were not scheduled
    INITIAL_SORTING_COLUMNS = ['glass_type', 'width', 'height']
    INITIAL_SORTING_ORDER = [True, True, True]
    TRIPLE_GLAZED_PANES = []
    ALL_TYPES = ['R6', 'R8']

    def __init__(self):
        """
        Initialize the scheduler with production data
        """
        self.production_plan_df = None

        self.current_record_num = 1

        self.num_of_shifts = None  # One or two shifts?
        self.total_sum_of_windows = 0
        self.middle_point = 0  # Middle point of the production plan - 'kreska', frozen part of the plan

        self.sum_of_triple_glazed = 0
        self.sum_of_efls = 0
        self.total_num_of_small_orders = 0
        self.total_num_of_first_and_last_positions_orders = 0

        self.unique_widths = list()  # Set to store unique widths of windows
        self.temp_unique_widths = list()  # Temporary set to store unique widths in third part planning
        self.last_width_index = 0  # Index of the last width in the unique widths list

        self.unique_sizes = list()  # Set to store unique sizes of windows
        self.temp_unique_sizes = list()  # Temporary set to store unique sizes in third part planning
        self.last_size_index = 0  # Index of the last size in the unique sizes list

        self.production_order_numbers_for_last_positions = []  # List to store production order numbers for first and last positions
        self.production_order_numbers_for_first_positions = []

        self.windows_types = self.ALL_TYPES
        self.color_before_middle_point = None  # Color of the windows with triple panes before the middle point

        # quantity of windows in first part of the production plan - triple glazed windows
        self.quantity_of_windows_in_first_part = 0

        # variables defining scheduling process
        self.sum_of_scheduled_orders = 0
        self.small_orders_sequence = 0  # Counter for small orders sequence
        self.can_be_small_order = True  # Flag to indicate if the order can be small
        self.ignore_small_sequence_condition = False  # Flag to ignore small orders sequence condition
        self.can_be_material_unavailable = False  # Flag to indicate if the order can be material unavailable
        self.possible_types = None  # Possible window types for scheduling
        self.quantity_scheduled_in_previous_iteration = 0  # Total Quantity of windows scheduled in the previous iteration (total scheduled sum at the end of the loop)
        self.number_of_empty_loops = 0  # Counter for empty loops in the scheduling process
        self.ignore_height_matching_condition = False  # Flag to ignore height matching condition
        self.ignore_width_matching_condition = False  # Flag to ignore width matching condition
        self.skip_widths_until_last_width = False  # Flag to skip widths until the last width in the unique widths list

        # variables defining last scheduled order
        self.last_order_width = None
        self.last_order_height = None
        self.last_order_size = None
        self.last_order_type = None
        self.last_order_glass = None
        self.last_order_color = None
        self.last_order_is_triple = None
        self.last_order_quantity = 0
        self.last_order_is_small = None
        self.last_order_prd_num = None
        self.last_order_product = None

        # TODO: Change debug mode Excel/Clipboard
        # load the data into a DataFrame
        # self.load_production_plan_excel()
        self.load_production_plan_zpp_cserie()

        # add new columns to the DataFrame based on various conditions
        self.is_small_order()
        self.is_triple()
        self.is_efl()
        self.fill_dummy_data()
        self.is_urgent_till_2_pm()
        self.is_urgent_till_6_pm()
        self.is_material_available()
        self.add_scheduling_columns()
        self.sort_production_plan()
        # self.gather_production_order_numbers_for_first_and_last_positions()

        # Calculate some statistics
        self.calculate_total_sum_of_windows()
        self.calculate_num_of_shifts()  # define num of shifts
        self.count_small_orders()
        self.calculate_middle_point()
        self.calculate_efls()
        self.get_unique_widths()


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
            "product",
            'window_type',
            'variant',
            'is_small',
            'is_triple',
            'is_efl',
            'is_urgent_till_2_pm',
            'is_urgent_till_6_pm',
            'is_material_available',
            'scheduling_position',
            'is_scheduled',
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
            "customer_order_number",
            "Poz. zlec. klienta",
            "goods_receiver",
            "prd_ord_num",
            "sap_nr",
            "product_name",
            "quantity",
            "Data dostępn. mat.",
            "long_text",
            "system_status",
            "glass_type",
            "profile_color",
            "width",
            "height",
            "Merkmalwert 02",
            "product",
            "window_type",
            "Merkmalwert 16",
            "Rozpoczęcie według harmonogramu",
            "Merkmalwert 01",
            "variant",
            "roller_blind",
            "Rodzaj zlecenia",
            "Merkmalwert 14",
            "Merkmalwert 20",
            "Merkmalwert 12",
            "Kontroler MRP",
            "Merkmalwert 25"
        ]

        headers_order = [
            # 'record_number',
            'customer_order_number',
            'goods_receiver',
            'prd_ord_num',
            'sap_nr',
            'product_name',
            'quantity',
            "long_text",
            'system_status',
            'glass_type',
            'profile_color',
            'width',
            'height',
            'window_type',
            'variant',
            'product',
            'roller_blind'
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
            by=self.INITIAL_SORTING_COLUMNS,
            ascending=self.INITIAL_SORTING_ORDER, inplace=True)

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

    def is_efl(self):
        """
        Check if the order is service sash window order
        :return:
        """
        self.production_plan_df['is_efl'] = self.production_plan_df['product_name'].str.contains('EFL')

    def fill_dummy_data(self):
        mask = (self.production_plan_df['product_name'].str.contains('EFL', na=False)) & (
                    self.production_plan_df['product'].isna() | (self.production_plan_df['product'] == ''))

        self.production_plan_df.loc[mask, 'product'] = 'EFL'

    def calculate_triple_glazed(self):
        """
        Calculate the number of triple-glazed windows in the production plan
        """
        self.sum_of_triple_glazed = self.production_plan_df[self.production_plan_df['is_triple']]['quantity'].sum()

    def calculate_total_sum_of_windows(self):
        """
        Calculate the total sum of windows in the production plan
        """
        self.total_sum_of_windows = self.production_plan_df['quantity'].sum()

    def calculate_efls(self):
        self.sum_of_efls = self.production_plan_df[self.production_plan_df['product'] == 'EFL']['quantity'].sum()
        print(f"EFL sum: {self.sum_of_efls}")

    def count_small_orders(self):
        """
        Count the number of small orders in the production plan
        """
        self.total_num_of_small_orders = self.production_plan_df[self.production_plan_df['is_small']].shape[0]

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
        # TODO: Usunąć/dodać ZTCH do testów
        self.production_plan_df['is_material_available'] = self.production_plan_df['system_status'].str.startswith(
            ('ZWOL'))

    def get_unique_widths(self):
        """
        Get unique widths of windows in the production plan
        """
        self.unique_widths = list(set(self.production_plan_df['width'].unique()))
        self.unique_widths.sort()  # Sort the unique widths for better readability

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

        print('Middle point: ', self.middle_point)

    def add_scheduling_columns(self):
        """
        Add a column to the DataFrame indicating the scheduling position of each order
        False means that the order is not scheduled yet
        1 means that the order is scheduled for the first position and so on
        """
        self.production_plan_df['scheduling_position'] = None  # Initialize with None
        self.production_plan_df['is_scheduled'] = False  # Initialize with False

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

    def schedule_one_position_basic(self, df_row):
        """
        Schedule one position in the production plan
        """
        self.production_plan_df.at[df_row.Index, 'scheduling_position'] = self.current_record_num
        self.production_plan_df.at[df_row.Index, 'is_scheduled'] = True
        self.current_record_num += 1

        # Update last scheduled order details
        self.last_order_width = df_row.width
        self.last_order_height = df_row.height
        self.last_order_is_triple = df_row.is_triple
        self.last_order_is_small = df_row.is_small
        self.last_order_type = df_row.window_type
        self.last_order_glass = df_row.glass_type
        self.last_order_color = df_row.profile_color
        self.last_order_quantity = df_row.quantity
        self.last_order_prd_num = df_row.prd_ord_num
        self.last_width_index = self.unique_widths.index(df_row.width)
        self.last_order_product = df_row.product

        self.sum_of_scheduled_orders += df_row.quantity
        print(f"Window scheduled: {df_row.product_name} Quantity: {df_row.quantity}")
        print("sum of scheduled orders:", self.sum_of_scheduled_orders)

        self.reset_empty_loops_counter()  # Reset the empty loops counter after scheduling an order

    def schedule_one_position_additional(self, df_row):
        self.handle_small_orders_sequence(df_row.is_small)
        self.handle_material_availability()

    def schedule_production_plan(self):
        """
        Schedule the second part of the production plan - Double glazed windows
        """
        self.quantity_scheduled_in_previous_iteration = 0

        is_planning_finished = False
        is_first_iteration = True

        self.possible_types = self.windows_types

        print("Starting production plan with possible types:", self.possible_types)
        steps = len(self.unique_widths) * 150
        iterator = self.ping_pong_iter(self.unique_widths, start_index=self.last_width_index, steps=steps)
        counter = 0  # Counter for the number of iterations
        loop_counter = 0  # Counter for the number of loops
        width = next(iterator)

        repeat_iteration_over_df = False  # Flag to indicate if we need to repeat the iteration over the DataFrame

        # Schedule production plan
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

                # Check if all windows are scheduled
                windows_left = self.production_plan_df[
                    (~self.production_plan_df['is_scheduled'])
                    ]['quantity'].sum()
                if windows_left == 0:
                    is_planning_finished = True
                    break

                if row.is_scheduled:
                    continue
                if row.prd_ord_num in self.production_order_numbers_for_last_positions:
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
                if row.width == width:
                    self.schedule_one_position_basic(row)
                    self.schedule_one_position_additional(row)
                    if not self.last_order_is_small:
                        repeat_iteration_over_df = True  # Set the flag to repeat the iteration over the DataFrame
                        break

            if divmod(counter, len(self.unique_widths))[1] == 0 and counter != 0 and not repeat_iteration_over_df:
                # If we have iterated through all unique widths, check if it wasn't empty loop
                # if it was empty loop, we need to switch off small orders sequence
                loop_counter += 1
                # print(f"Loop counter: {loop_counter}")
                self.empty_loop_check()

            if is_planning_finished:
                print("Planning finished.")
                break

    def start_or_finish_the_production_plan(self, num_of_orders_to_plan):
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
                if row.prd_ord_num in self.production_order_numbers_for_last_positions:
                    self.schedule_one_position_basic(row)
                    self.schedule_one_position_additional(row)
                    counter += 1
                    if counter >= num_of_orders_to_plan:
                        return  # Exit the function if we reached the number of orders to plan
            self.empty_loop_check()  # Check if the loop was empty after each iteration

    def main_scheduling_function(self):
        """
        Schedule production orders based on the production plan and defined conditions
        """
        self.schedule_production_plan()
        self.copy_df_index_to_clipboard(column_name='scheduling_position', new_col_name='copy_pos')
        self.display_view()