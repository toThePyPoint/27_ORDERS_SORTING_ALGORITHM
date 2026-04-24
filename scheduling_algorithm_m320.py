import pandas as pd
import re


from scheduling_algorithm_basic import ProductionOrderSchedulerBasic


class ProductionOrderSchedulerM320(ProductionOrderSchedulerBasic):
    ALL_TYPES = ['Q4']
    INITIAL_SORTING_COLUMNS = ['is_small', 'glass_type', 'width', 'height']
    INITIAL_SORTING_ORDER = [False, True, True, True]
    MIDDLE_POINT_PROPORTION = 0.6
    TWO_SHIFTS_THRESHOLD = 240
    SMALL_ORDERS_MAX_SEQUENCE = 30
    UNTYPICAL_HEIGHTS_AFTER_MIDDLE_POINT = [70, 50]
    UNTYPICAL_WIDTHS_AFTER_MIDDLE_POINT = [70, 50]

    def __init__(self):
        super().__init__()
        self.possible_products = None
        self.last_position_width = None
        self.last_position_order_number = None

        self.is_dummy_allowed = False
        self.starting_plan = True
        self.finishing_plan = False
        self.finishing_plan_last_position = False
        self.switched_to_last_type = False
        self.skip_last_order_width = False
        self.increased_last_orders_list = False
        self.untypical_heights_allowed = False

        self.starting_orders_scheduled = 0
        self.quantity_of_first_type_sequence = 0

        self.ignore_untypical_sizes_condition = False

        self.width_map = {
            '05': 540,
            '06': 650,
            '07': 740,
            '09': 940,
            '11': 1140,
            '13': 1340
            # dodaj kolejne...
        }

        self.height_map = {
            '07': 780,
            '09': 980,
            '11': 1180,
            '14': 1400,
            '16': 1600,
            # dodaj kolejne...
        }

        self.fill_dummy_data()
        self.fill_dimensions()

         # Repeat these functions in child class
        self.sort_production_plan()
        self.get_unique_widths()
        self.calculate_middle_point()
        self.gather_production_order_numbers_for_first_and_last_positions()
        self.num_of_first_positions = len(self.production_order_numbers_for_first_positions)
        self.num_of_last_positions = len(self.production_order_numbers_for_last_positions)

    def fill_dummy_data(self):
        # Definiujemy mapowanie: co szukamy -> do której kolumny -> jaka wartość
        mappings = [
            ('EFL', 'product', 'EFL'),
            ('435', 'window_type', '435'),
            ('439', 'window_type', '439'),
            ('735', 'window_type', '735'),
        ]

        self.production_plan_df['is_dummy'] = False

        for search_text, target_col, fill_value in mappings:
            # Tworzymy maskę dynamicznie dla każdej pary
            mask = (
                    self.production_plan_df['product_name'].str.contains(search_text, na=False) &
                    (self.production_plan_df[target_col].isna() | (self.production_plan_df[target_col] == ''))
            )

            self.production_plan_df.loc[mask, target_col] = fill_value

        # --- Nowa część: Ekstrakcja danych z 'long_text' dla 'ARTIKEL' ---

        # 1. Definiujemy maskę dla wierszy zawierających 'ARTIKEL'
        artikel_mask = self.production_plan_df['product_name'].str.contains('ARTIKEL', case=False, na=False)

        # 2. Wyrażenie regularne do wyciągnięcia: R6, 9G, 134, 140
        # Grupy: (R6)(9G) (134)/(140)
        regex_pattern = r'([A-Z]\d)(\d[A-Z_]?)\s+(\d+)/(\d+)'

        def extract_and_fill(row):
            text = str(row['long_text'])
            match = re.search(regex_pattern, text)

            if match:
                w_type, g_type, w_val, h_val = match.groups()

                # Uzupełniamy tylko jeśli pole jest puste/NaN
                if not row.get('window_type') or pd.isna(row['window_type']):
                    row['window_type'] = w_type

                if not row.get('glass_type') or pd.isna(row['glass_type']):
                    row['glass_type'] = g_type

                if not row.get('width') or pd.isna(row['width']):
                    row['width'] = int(w_val) * 10

                if not row.get('height') or pd.isna(row['height']):
                    row['height'] = int(h_val) * 10

                if not row.get('product') or pd.isna(row['product']):
                    row['product'] = 'EFL' if 'EFL' in text else 'WDF'

                row['is_dummy'] = True

            return row

        # Aplikujemy funkcję tylko do przefiltrowanego podzbioru danych
        self.production_plan_df.loc[artikel_mask] = self.production_plan_df[artikel_mask].apply(extract_and_fill,
                                                                                                axis=1)

    def fill_dimensions(self):
        # 1. Niezależne mapowania dla szerokości (pierwszy człon) i długości (drugi człon)


        def extract_and_map(row):
            # Pobieramy kody z nazwy produktu
            match = re.search(r'(\d{2})/(\d{2})', str(row['product_name']))

            current_w = row['width']
            current_l = row['height']

            if match:
                w_code, l_code = match.groups()

                # Uzupełnij szerokość tylko jeśli jest pusta
                if pd.isna(current_w) or current_w == '':
                    current_w = self.width_map.get(w_code, current_w)

                # Uzupełnij długość tylko jeśli jest pusta
                if pd.isna(current_l) or current_l == '':
                    current_l = self.height_map.get(l_code, current_l)

            return current_w, current_l

        # Zastosowanie transformacji
        if not self.production_plan_df.empty:
            self.production_plan_df[['width', 'height']] = self.production_plan_df.apply(
                extract_and_map, axis=1, result_type='expand'
            )

    def handle_starting_and_finishing_plan(self):
        if self.last_order_prd_num in self.production_order_numbers_for_first_positions:
            self.starting_orders_scheduled += 1

        if self.starting_orders_scheduled >= len(self.production_order_numbers_for_first_positions):
            self.starting_plan = False

    def schedule_production_plan(self):
        """
        Schedule the second part of the production plan - Double glazed windows
        """
        self.quantity_scheduled_in_previous_iteration = 0

        is_planning_finished = False
        is_first_iteration = True

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
                # If we dropped height condition, we want to come back to last scheduled width so that the width consistency is kept if possible
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
                # if not self.finishing_plan:
                #     if row.prd_ord_num in self.production_order_numbers_for_last_positions:
                #         continue
                # if not self.finishing_plan_last_position:
                #     if row.prd_ord_num == self.last_position_order_number:
                #         continue
                if self.starting_plan:
                    if row.prd_ord_num not in self.production_order_numbers_for_first_positions:
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
                if row.is_dummy and not self.is_dummy_allowed:
                    continue
                if not self.untypical_heights_allowed and not self.ignore_untypical_sizes_condition:
                    if row.height in self.UNTYPICAL_HEIGHTS_AFTER_MIDDLE_POINT:
                        continue
                    if row.width in self.UNTYPICAL_WIDTHS_AFTER_MIDDLE_POINT:
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

    def schedule_production_plan_last_position(self):
        """
        Schedule last position
        """
        self.quantity_scheduled_in_previous_iteration = 0

        print("Starting last position planning:", self.possible_types)

        for row in self.production_plan_df.itertuples():

            if row.is_scheduled:
                continue
            if row.prd_ord_num in self.production_order_numbers_for_last_positions:
                self.schedule_one_position_basic(row)
                self.schedule_one_position_additional(row)

    def schedule_one_position_additional(self, df_row):
        self.handle_small_orders_sequence(df_row.is_small)
        self.handle_material_availability()
        self.handle_starting_and_finishing_plan()
        self.handle_dummy_orders()
        self.handle_untypical_heights_and_widths()

    def gather_production_order_numbers_for_first_and_last_positions(self):
        """
        Gather production order numbers for first and last positions in the production plan
        """
        self.production_order_numbers_for_first_positions = self.production_plan_df[
            (self.production_plan_df['is_material_available']) &
            (~self.production_plan_df['width'].isin(self.UNTYPICAL_WIDTHS_AFTER_MIDDLE_POINT)) &
            (~self.production_plan_df['height'].isin(self.UNTYPICAL_HEIGHTS_AFTER_MIDDLE_POINT))
        ].sort_values(
            by=['is_urgent_till_2_pm', 'is_urgent_till_6_pm', 'customer_order_number'],
            ascending=[False, False, True],
            na_position='last'
        )['prd_ord_num'].head(1).tolist()

        print("First positions: ", self.production_order_numbers_for_first_positions)

    def main_scheduling_function(self):
        """
        Schedule production orders based on the production plan and defined conditions
        """
        self.schedule_production_plan()
        # self.schedule_production_plan_last_position()
        self.copy_df_index_to_clipboard(column_name='scheduling_position', new_col_name='copy_pos')
        self.display_view()

    def handle_dummy_orders(self):
        """
        Can be scheduled only after middle point
        :return:
        """
        if self.sum_of_scheduled_orders >= self.middle_point:
            self.is_dummy_allowed = True

    def handle_untypical_heights_and_widths(self):
        if self.sum_of_scheduled_orders >= self.middle_point:
            self.untypical_heights_allowed = True

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
            self.ignore_untypical_sizes_condition = True

        if self.number_of_empty_loops >= 10:
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
        self.ignore_untypical_sizes_condition = False
        self.number_of_empty_loops = 0  # Reset the counter for empty loops


