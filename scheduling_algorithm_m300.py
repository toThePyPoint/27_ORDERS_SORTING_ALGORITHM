import pandas as pd
import re


from scheduling_algorithm_basic import ProductionOrderSchedulerBasic


class ProductionOrderSchedulerM300(ProductionOrderSchedulerBasic):
    ALL_TYPES = ['R6', 'R8', 'R8S', '627', '847']
    OLD_GEN_TYPES = ['627', '847']
    NEW_GEN_TYPES = ['R6', 'R8', 'R8S']
    ALL_PRODUCTS = ['WDF', 'WDT', 'WSA', 'EFL']
    INITIAL_SORTING_COLUMNS = ['glass_type', 'width', 'height']
    INITIAL_SORTING_ORDER = [True, True, True]
    SMALL_ORDERS_MAX_SEQUENCE = 30
    MIDDLE_POINT_PROPORTION = 0.6
    TWO_SHIFTS_THRESHOLD = 180

    def __init__(self):
        super().__init__()
        self.sum_of_efls_627 = 0
        self.sum_of_efls_847 = 0
        self.possible_windows_before_middle_point = 0
        self.possible_sashes_before_middle_point = 0
        self.last_production_order_quantity = 0  # quantity of last production order

        self.possible_products = None

        self.is_dummy_allowed = False

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
        self.calculate_labor_coefficients()
        self.calculate_middle_point()
        self.calculate_efls()
        self.calculate_products_before_middle_point()
        self.select_last_order()

    def fill_dummy_data(self):
        # Definiujemy mapowanie: co szukamy -> do której kolumny -> jaka wartość
        mappings = [
            ('EFL', 'product', 'EFL'),
            ('627', 'window_type', '627'),
            ('847', 'window_type', '847'),
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
        # regex_pattern = r'([A-Z]\d)(\d[A-Z])\s+(\d+)/(\d+)' # old
        regex_pattern = r'([A-Z]\d)(\d[A-Z_]?)\s+(\d+)/(\d+)' # improved

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

    def calculate_labor_coefficients(self):
        def get_labor_factor(row):
            # 1. Sprawdzamy rolety (najwyższy wskaźnik/priorytet)
            if row['roller_blind'] in ['ZAR', 'ZRV']:
                return 1.5

            # 2. Sprawdzamy typ produktu
            val = row['product']
            if val == 'WDT':
                return 1.5
            if val == 'EFL':
                return 0.5

            # 3. Wartość domyślna (jeśli żaden warunek nie jest spełniony)
            return 1.0

        # Obliczamy nową kolumnę (opcjonalnie) i aktualizujemy quantity lub tworzymy labor_time
        if not self.production_plan_df.empty:
            factors = self.production_plan_df.apply(get_labor_factor, axis=1)
            self.production_plan_df['quantity'] = self.production_plan_df['quantity'] * factors

    def calculate_efls(self):
        self.sum_of_efls = self.production_plan_df[(self.production_plan_df['product'] == 'EFL') &
                                                   (self.production_plan_df['window_type'].isin(('R6', 'R8')))][
            'quantity'].sum()
        self.sum_of_efls_847 = self.production_plan_df[(self.production_plan_df['product'] == 'EFL') &
                                                       (self.production_plan_df['window_type'].isin(('847',)))][
            'quantity'].sum()
        self.sum_of_efls_627 = self.production_plan_df[(self.production_plan_df['product'] == 'EFL') &
                                                       (self.production_plan_df['window_type'].isin(('627',)))][
            'quantity'].sum()
        print(f"EFL sum: {self.sum_of_efls}")
        print(f"EFL 847 sum: {self.sum_of_efls_847}")
        print(f"EFL 627 sum: {self.sum_of_efls_627}")

    def calculate_products_before_middle_point(self):
        self.possible_windows_before_middle_point = \
        self.production_plan_df[(self.production_plan_df['is_material_available']) &
                                (self.production_plan_df['product'].isin(('WDT', 'WDF', 'WSA')))]['quantity'].sum()
        self.possible_sashes_before_middle_point = \
        self.production_plan_df[(self.production_plan_df['is_material_available']) &
                                (self.production_plan_df['product'] == "EFL")]['quantity'].sum()
        print(f'Possible windows before middle point: {self.possible_windows_before_middle_point}')
        print(f"Possible sashes before middle point: {self.possible_sashes_before_middle_point}")

    def handle_efls(self):
        efl_left = self.production_plan_df[(self.production_plan_df['product'] == 'EFL') &
                                           (~self.production_plan_df['is_scheduled'])]['quantity'].sum()
        efl_old_gen_left = self.production_plan_df[(self.production_plan_df['product'] == 'EFL') &
                                                   (self.production_plan_df['window_type'].isin(self.OLD_GEN_TYPES)) &
                                                   (~self.production_plan_df['is_scheduled'])]['quantity'].sum()
        efl_r6_r8_left = efl_left - efl_old_gen_left

        if self.sum_of_scheduled_orders >= self.middle_point or self.sum_of_scheduled_orders >= self.possible_windows_before_middle_point - self.last_production_order_quantity:
            self.possible_products = self.ALL_PRODUCTS

        if self.last_order_product == 'EFL' and efl_left:
            self.possible_products = ['EFL']

            if self.last_order_type in self.OLD_GEN_TYPES:
                if efl_old_gen_left:
                    self.possible_types = self.OLD_GEN_TYPES
                else:
                    self.possible_types = self.NEW_GEN_TYPES

            elif self.last_order_type in self.NEW_GEN_TYPES:
                if efl_r6_r8_left:
                    self.possible_types = self.NEW_GEN_TYPES
                else:
                    self.possible_types = self.OLD_GEN_TYPES

        if not efl_left:
            self.possible_products = self.ALL_PRODUCTS
            self.possible_types = self.ALL_TYPES

    def handle_dummy_orders(self):
        """
        Can be scheduled only after middle point
        :return:
        """
        if self.sum_of_scheduled_orders >= self.middle_point:
            self.is_dummy_allowed = True

    def schedule_production_plan(self):
        """
        Schedule the second part of the production plan - Double glazed windows
        """
        self.quantity_scheduled_in_previous_iteration = 0

        is_planning_finished = False
        is_first_iteration = True

        self.possible_types = self.windows_types
        self.possible_products = ['WDF', 'WDT', 'WSA']

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
        self.handle_efls()
        self.handle_dummy_orders()

    def select_last_order(self):
        last_ord_max_quantity = self.possible_windows_before_middle_point + self.possible_sashes_before_middle_point - self.middle_point
        last_ord_max_quantity = min(last_ord_max_quantity, 6)
        temp_df = self.production_plan_df[(self.production_plan_df['product'] != 'EFL') &
                                          (~self.production_plan_df['is_urgent_till_6_pm']) &
                                          (self.production_plan_df['quantity'] <= last_ord_max_quantity) &
                                          (~self.production_plan_df['roller_blind'].isin(('ZAR', 'ZRV'))) &
                                          (self.production_plan_df['customer_order_number'].isna())].sort_values(
            by='quantity', ascending=False)

        # if temp_df.empty:
        #     temp_df = self.production_plan_df[(self.production_plan_df['product'] != 'EFL') &
        #                                       (~self.production_plan_df['is_urgent_till_6_pm']) &
        #                                       (~self.production_plan_df['roller_blind'].isin(('ZAR', 'ZRV'))) &
        #                                       (self.production_plan_df[
        #                                            'quantity'] <= last_ord_max_quantity)].sort_values(
        #         by='quantity', ascending=False)

        # Sprawdzamy, czy temp_df nie jest pusty, aby uniknąć błędu IndexError
        if not temp_df.empty:
            order_num = temp_df['prd_ord_num'].iloc[0]
            self.production_order_numbers_for_last_positions.append(order_num)
            self.last_production_order_quantity = temp_df['quantity'].iloc[0]

        print(self.production_order_numbers_for_last_positions)

    def main_scheduling_function(self):
        """
        Schedule production orders based on the production plan and defined conditions
        """
        self.schedule_production_plan()
        if len(self.production_order_numbers_for_last_positions) > 0:
            self.schedule_production_plan_last_position()
        self.copy_df_index_to_clipboard(column_name='scheduling_position', new_col_name='copy_pos')
        self.display_view()