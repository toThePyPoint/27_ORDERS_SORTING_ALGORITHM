import pandas as pd
import re


from scheduling_algorithm_basic import ProductionOrderSchedulerBasic


class ProductionOrderSchedulerM200(ProductionOrderSchedulerBasic):
    ALL_TYPES = ['R3', 'R4', 'R5', 'R7', '435', '439']
    OLD_GEN_TYPES = ['435', '439', '735']
    NEW_GEN_TYPES = ['R3', 'R4', 'R5', 'R7']
    ALL_PRODUCTS = ['WDF', 'WDT', 'WDA', 'WRA', 'EFL']
    INITIAL_SORTING_COLUMNS = ['is_small', 'glass_type', 'width', 'height']
    INITIAL_SORTING_ORDER = [False, True, True, True]
    MIDDLE_POINT_PROPORTION = 0.6
    TWO_SHIFTS_THRESHOLD = 240

    def __init__(self):
        super().__init__()
        self.possible_products = None
        self.type_to_start_with = None # either R4 or R7
        self.type_to_end_with = None # either R4 or R7

        self.is_dummy_allowed = False
        self.starting_plan = True
        self.finishing_plan = False
        self.first_type_switched = False

        self.starting_orders_scheduled = 0
        self.finishing_orders_scheduled = 0
        self.quantity_of_first_type_sequence = 0
        self.r3_possible_before_middle_point = 0
        self.r4_possible_before_middle_point = 0
        self.r5_possible_before_middle_point = 0
        self.r7_possible_before_middle_point = 0

        self.r4_total_sum = 0
        self.r7_total_sum = 0

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

        self.fill_dimensions()

        # Repeat these functions in child class
        self.sort_production_plan()
        self.get_unique_widths()
        self.calculate_middle_point()
        self.add_is_kf_column()
        self.count_quantities_of_each_type_before_middle_point()
        self.count_r3_r4_r5_r7_orders()
        self.determine_type_to_start_with_and_end_with()
        self.determine_quantity_of_first_type_sequence()
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
        artikel_mask = self.production_plan_df['product_name'].str.contains('ARTIKEL', na=False)

        # 2. Wyrażenie regularne do wyciągnięcia: R6, 9G, 134, 140
        # Grupy: (R6)(9G) (134)/(140)
        regex_pattern = r'([A-Z]\d)(\d[A-Z])\s+(\d+)/(\d+)'

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

    def handle_dummy_orders(self):
        """
        Can be scheduled only after middle point
        :return:
        """
        if self.sum_of_scheduled_orders >= self.middle_point:
            self.is_dummy_allowed = True

    def count_r3_r4_r5_r7_orders(self):
        """
        Count the quantity of each type
        """
        self.r4_total_sum = self.production_plan_df[self.production_plan_df['window_type'] == 'R4']['quantity'].sum()
        self.r7_total_sum = self.production_plan_df[self.production_plan_df['window_type'] == 'R7']['quantity'].sum()

        print(f"Total sum of R4: {self.r4_total_sum}")
        print(f"Total sum of R7: {self.r7_total_sum}")

    def handle_window_type(self):
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

        if self.sum_of_scheduled_orders >= self.quantity_of_first_type_sequence and not self.first_type_switched:
            if self.type_to_start_with == 'R7':
                self.possible_types = ['R4']
                print(f"Switch type to {self.possible_types}")
            else:
                self.possible_types = ['R7']
                print(f"Switch type to {self.possible_types}")
            self.first_type_switched = True
        # If the sum of scheduled orders reaches the quantity of windows in the first part minus R39, force R39 type
        # r3_trigger = self.sum_of_scheduled_orders >= self.quantity_of_windows_in_first_part - self.r3_possible_before_middle_point
        #
        # elif planning_mode == self.planning_modes.second_part_one_shift:
        #     # plan r3 when other doubled glazed windows are planned and there are any r3 double
        #     if self.can_be_material_unavailable:
        #         windows_left = self.production_plan_df[
        #             (~self.production_plan_df['is_scheduled']) &
        #             (self.production_plan_df['window_type'] != 'R3') &
        #             (~self.production_plan_df['is_triple'])
        #             ]['quantity'].sum()
        #     else:
        #         windows_left = self.production_plan_df[
        #             (~self.production_plan_df['is_scheduled']) &
        #             (self.production_plan_df['window_type'] != 'R3') &
        #             (~self.production_plan_df['is_triple']) &
        #             (self.production_plan_df['is_material_available'])
        #             ]['quantity'].sum()
        #     r3_trigger = self.sum_of_r3_double > 0 and windows_left == 0
        # elif planning_mode == self.planning_modes.third_part_one_shift:
        #     # plan r3 if last order type was R3 and there are any r3 triple in production plan
        #     r3_trigger = self.last_order_type == 'R3' and self.sum_of_r3_triple > 0
        # else:
        #     r3_trigger = None
        #
        # if r3_trigger:
        #     print(f"R3 left:", r3_left)
        #     if r3_left > 0:
        #         self.possible_types = ['R3']
        #         self.force_type = True
        #     else:
        #         self.possible_types = self.windows_types
        #         self.force_type = False

    def handle_starting_and_finishing_plan(self):
        if self.last_order_prd_num in self.production_order_numbers_for_first_positions:
            self.starting_orders_scheduled += 1
        if self.last_order_prd_num in self.production_order_numbers_for_last_positions:
            self.finishing_orders_scheduled += 1

        if self.starting_orders_scheduled >= len(self.production_order_numbers_for_first_positions):
            self.starting_plan = False

        # TODO: Implement finishing the plan

    def schedule_production_plan(self):
        """
        Schedule the second part of the production plan - Double glazed windows
        """
        self.quantity_scheduled_in_previous_iteration = 0

        is_planning_finished = False
        is_first_iteration = True

        self.possible_types = [self.type_to_start_with]
        self.possible_products = ['WDF', 'WDT', 'EFL']

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
                if not self.finishing_plan:
                    if row.prd_ord_num in self.production_order_numbers_for_last_positions:
                        continue
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
                if not row.product in self.possible_products:
                    continue
                if not row.window_type in self.possible_types:
                    continue
                if row.is_dummy and not self.is_dummy_allowed:
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
        self.handle_dummy_orders()
        self.handle_starting_and_finishing_plan()
        self.handle_window_type()
        print(f"Window scheduled: {df_row.product_name} Quantity: {df_row.quantity}")

    def gather_production_order_numbers_for_first_and_last_positions(self):
        """
        Gather production order numbers for first and last positions in the production plan
        """
        self.production_order_numbers_for_first_positions = self.production_plan_df[
            (self.production_plan_df['quantity'] >= 12) &
            (self.production_plan_df['is_material_available']) &
            (self.production_plan_df['window_type'] == self.type_to_start_with)
        ].sort_values(
            by=['is_urgent_till_6_pm', 'customer_order_number'],
            ascending=[False, True],
            na_position='last'
        )['prd_ord_num'].head(1).tolist()

        self.production_order_numbers_for_last_positions = self.production_plan_df[
            (self.production_plan_df['quantity'] >= 12) &
            (~self.production_plan_df['prd_ord_num'].isin(self.production_order_numbers_for_first_positions)) &
            (~self.production_plan_df['is_urgent_till_6_pm']) &
            (self.production_plan_df['window_type'] == self.type_to_end_with)
        ].sort_values(
            by=['customer_order_number'],
            ascending=[False],
            na_position='last'
        )['prd_ord_num'].head(1).tolist()

        print(f"First order: {self.production_plan_df.loc[self.production_plan_df['prd_ord_num'] == self.production_order_numbers_for_first_positions[0], 'product_name'].item()}")
        print(f"Last order: {self.production_plan_df.loc[self.production_plan_df['prd_ord_num'] == self.production_order_numbers_for_last_positions[0], 'product_name'].item()}")

    def main_scheduling_function(self):
        """
        Schedule production orders based on the production plan and defined conditions
        """
        self.schedule_production_plan()
        # self.schedule_production_plan_last_position()
        self.copy_df_index_to_clipboard(column_name='scheduling_position', new_col_name='copy_pos')
        self.display_view()

    def count_quantities_of_each_type_before_middle_point(self):
        self.r4_possible_before_middle_point = \
        self.production_plan_df[(self.production_plan_df['window_type'] == 'R4') &
                                (self.production_plan_df['is_material_available']) &
                                ((self.production_plan_df['profile_color'] != 'WH') | (
                                            self.production_plan_df['is_KF'] == True))
                                ]['quantity'].sum()

        self.r7_possible_before_middle_point = \
        self.production_plan_df[(self.production_plan_df['window_type'] == 'R7') &
                                (self.production_plan_df['is_material_available']) &
                                ((self.production_plan_df['profile_color'] != 'WH') | (
                                            self.production_plan_df['is_KF'] == True))
                                ]['quantity'].sum()

        self.r3_possible_before_middle_point = self.production_plan_df[
            (self.production_plan_df['window_type'] == 'R3') & (self.production_plan_df['is_material_available']) &
            (self.production_plan_df['profile_color'] != 'WH')]['quantity'].sum()

    def determine_type_to_start_with_and_end_with(self):
        if self.r4_total_sum < self.middle_point:
            self.type_to_start_with = 'R7'
            self.type_to_end_with = 'R7'
        elif self.r4_total_sum > 0.7 * self.total_sum_of_windows:
            self.type_to_start_with = 'R4'
            self.type_to_end_with = 'R4'
        else:
            self.type_to_start_with = 'R4'
            self.type_to_end_with = 'R7'

        print("Type to start with is", self.type_to_start_with)
        print("Type to end with is", self.type_to_end_with)

    def determine_quantity_of_first_type_sequence(self):
        # TODO: continue here
        if self.type_to_start_with == 'R7':
            self.quantity_of_first_type_sequence = self.r7_possible_before_middle_point // 2

    def add_is_kf_column(self):
        self.production_plan_df['is_KF'] = self.production_plan_df.apply(lambda row: row['product_name'].endswith('KF'), axis=1)

