#------------------------------------------------------------------------------
#
# Author: Vinhthuy Phan, 2021
#
#------------------------------------------------------------------------------
import pandas
import numpy
import seaborn
import joblib

class Model:
    #--------------------------------------------------------------------------
    # data_file: file containing processed X and y
    # additional_file: file containing information about term, pidm, index
    #--------------------------------------------------------------------------
    def __init__(self, model_file, data_file, additional_file, merit_quantiles, need_quantiles, affordable_threshold):
        self.model = joblib.load(model_file)
        self.merit_quantiles = merit_quantiles
        self.need_quantiles = need_quantiles
        self.merit_buckets = None
        self.need_buckets = None
        self.affordable_threshold = affordable_threshold
        self.data_file = data_file
        self.additional_file = additional_file
        self.indices = {}
        self.add_df = pandas.read_csv(additional_file)
        for term, indices in self.add_df.groupby('term').groups.items():
            self.indices[term] = indices
            
        self.df = pandas.read_csv(data_file)
        self.y = self.df['enr_flag']
        self.stats = pandas.DataFrame()
        self.stats['achievement_index'] = self.df.act_composite_high + 3 * self.df.high_school_gpa
        self.stats['merit_cat'] = pandas.qcut(self.stats.achievement_index, q=self.merit_quantiles, labels=False)
        self.stats['enrollment'] = 1
        self.init_stats()
        self.original_full_stats = self.create_full_stats()
        self.baseline = self.compute_value(self.original_full_stats, self.y)

    #--------------------------------------------------------------------------
    
    def is_accessible(self, r):
        return (r['affordability']<self.affordable_threshold and r['amount_mn']>0).astype(int)


    #--------------------------------------------------------------------------

    def init_stats(self):
        self.X = self.df.drop(columns=['enr_flag'])
        self.y_prob = self.y.copy()
        self.stats['revenue'] = self.df.expected_tui - self.X.amount_merit - self.X.amount_need
        self.stats['amount_mn'] = self.X.amount_merit + self.X.amount_need
        # do this:
        # self.stats['roi'] = self.stats['revenue'] / self.stats['amount_mn']
        #
        self.stats['need_index'] = self.X.amount_merit + self.df.efc + self.df.est_annual_ft_pell - self.df.coa
        self.stats['need_cat'] = pandas.cut(self.stats.need_index, self.need_quantiles,labels=False,include_lowest=True)
        self.stats['affordability'] = self.stats['need_index'] + self.X['amount_need']
        # self.stats['accessible'] = self.stats.affordability.apply(
        #     lambda a: 0 if a<self.affordable_threshold else 1)
        self.stats['accessible'] = self.stats.apply(self.is_accessible, axis = 1)
        self.value = self.compute_value(self.stats, self.y)
        self.saved_value = self.value.copy()

    #--------------------------------------------------------------------------

    def save_value(self):
        self.saved_value = self.value.copy()

    #--------------------------------------------------------------------------
    # model: class Model instance (pre-built model)
    #--------------------------------------------------------------------------
    def apply_model(self):
        if self.X is None or self.stats is None:
            raise Exception('Attributes (X) must exist to compute value from model.')
        if self.model is None:
            raise Exception('There is no model.')
        self.y_prob = self.model.predict_proba(self.X)[:, 1]
        self.value = self.compute_value(self.stats, self.y_prob)
        # self.value['roi'] = self.value['revenue'] / self.value['amount_mn']


    #--------------------------------------------------------------------------
    def apply_need_merit_buckets(self, sol):
        if sol.merit_buckets is not None:
            self.merit_buckets = sol.merit_buckets.value
            self.X['amount_merit'] = self.stats.merit_cat.apply(sol.merit_buckets.lookup)
            self.stats['need_index'] = self.X['amount_merit'] + self.df.efc + self.df.est_annual_ft_pell - self.df.coa
            self.stats['need_cat'] = pandas.cut(self.stats['need_index'], self.need_quantiles,labels=False,include_lowest=True)
        if sol.need_buckets is not None:
            self.need_buckets = sol.need_buckets.value
            self.X['amount_need'] = self.stats['need_cat'].apply(sol.need_buckets.lookup)
        if sol.merit_buckets is not None or sol.need_buckets is not None:
            self.stats['revenue'] = self.df.expected_tui - self.X['amount_merit'] - self.X['amount_need']
            self.stats['amount_mn'] = self.X['amount_merit'] + self.X['amount_need']
            self.stats['affordability'] = self.stats['need_index'] + self.X['amount_need']
            # self.stats['accessible'] = self.stats['affordability'].apply(
            #     lambda a: 0 if a<self.affordable_threshold else 1)
            self.stats['accessible'] = self.stats.apply(self.is_accessible, axis = 1)
            self.apply_model()

    #--------------------------------------------------------------------------
    def apply_need_buckets(self, sol):
        if sol.need_buckets is not None:
            self.need_buckets = sol.need_buckets.value
            self.X['amount_need'] = self.stats['need_cat'].apply(sol.need_buckets.lookup)
            self.stats['revenue'] = self.df.expected_tui - self.X.amount_merit - self.X['amount_need']
            self.stats['amount_mn'] = self.X['amount_merit'] + self.X['amount_need']
            self.stats['affordability'] = self.stats['need_index'] + self.X['amount_need']
            # self.stats['accessible'] = self.stats['affordability'].apply(
            #     lambda a: 0 if a<self.affordable_threshold else 1)
            self.stats['accessible'] = self.stats.apply(self.is_accessible, axis = 1)
            self.apply_model()

    #--------------------------------------------------------------------------
    # Note: changes amount_merit may change to amount_need
    #--------------------------------------------------------------------------
    def apply_merit_buckets(self, sol):
        if sol.merit_buckets is not None:
            self.merit_buckets = sol.merit_buckets.value
            self.X['amount_merit'] = self.stats.merit_cat.apply(sol.merit_buckets.lookup)
            self.stats['need_index'] = self.X['amount_merit'] + self.df.efc + self.df.est_annual_ft_pell - self.df.coa
            self.stats['need_cat'] = pandas.cut(self.stats['need_index'], self.need_quantiles,labels=False,include_lowest=True)
            if sol.need_buckets is not None:
                self.X['amount_need'] = self.stats['need_cat'].apply(sol.need_buckets.lookup)
            self.stats['revenue'] = self.df.expected_tui - self.X['amount_merit'] - self.X['amount_need']
            self.stats['amount_mn'] = self.X['amount_merit'] + self.X['amount_need']
            self.stats['affordability'] = self.stats['need_index'] + self.X['amount_need']
            # self.stats['accessible'] = self.stats['affordability'].apply(
            #     lambda a: 0 if a<self.affordable_threshold else 1)
            self.stats['accessible'] = self.stats.apply(self.is_accessible, axis = 1)
            self.apply_model()

    #--------------------------------------------------------------------------
    def create_full_stats(self):
        full_stats = self.stats.copy()
        full_stats['act_composite_high'] = self.X['act_composite_high']
        full_stats['high_school_gpa'] = self.X['high_school_gpa']
        full_stats['amount_merit'] = self.X['amount_merit']
        full_stats['amount_need'] = self.X['amount_need']
        return full_stats
    
    #--------------------------------------------------------------------------
    def compute_value(self, stats, probs, indices=None):
        if indices is None:
            u = stats.T
            v = probs
        else:
            u = stats.loc[indices].T
            v = probs[indices]
        result = u.dot(v)
        if type(result) == pandas.Series:
            result['roi'] = result['revenue'] / result['amount_mn']
            result['affordable'] = result['affordability'] / result['enrollment']
            result['achievement'] = result['achievement_index'] / result['enrollment']
        return result
        
    #--------------------------------------------------------------------------
    def compare_against_baseline(self, by_terms=False, use_probabilities=True):
        if self.value is None:
            print(None)
        else:
            full_stats = self.create_full_stats()
            if use_probabilities:
                value = self.compute_value(full_stats, self.y_prob)
            else:
                y_hat = self.model.predict(self.X)
                value = self.compute_value(full_stats, y_hat)

            print('Baseline (actual enrollment) vs. Current value (expected enrollment)')
            self._against_baseline(value, self.baseline)
            if by_terms:
                for term, idx in self.indices.items():
                    print('\nTerm:', term)
                    v = self.compute_value(full_stats, self.y_prob, idx)
                    b = self.compute_value(full_stats, self.y, idx)
                    self._against_baseline(v, b)
    
            if self.merit_buckets is not None and self.need_buckets is not None:
                table = self.compare_against_baseline_per_category()
                return table

    #--------------------------------------------------------------------------
    def _against_baseline(self, value, baseline):
        print('Amount merit:        {}% (${})'.format(
            round(100*(value.amount_merit/baseline.amount_merit - 1),1), 
            round(value.amount_merit - baseline.amount_merit,0)))
        print('Amount need:         {}% (${})'.format(
            round(100*(value.amount_need/baseline.amount_need - 1), 1),
            round(value.amount_need - baseline.amount_need,0)))
        print('Amount merit+need:   {}% (${})'.format(
            round(100*(value.amount_mn/baseline.amount_mn - 1),1), 
            round(value.amount_mn - baseline.amount_mn,0)))

        print('Enrollment:          {}%'.format(round(100*(value.enrollment/baseline.enrollment - 1), 1)))
        print('Revenue:             {}% (${})'.format(
            round(100*(value.revenue/baseline.revenue - 1), 1),
            round(value.revenue - baseline.revenue,0)))
        print('Achievement index:   {}%'.format(
            round(100*(value.achievement_index/baseline.achievement_index - 1), 1)))
        print('Achievement:         {}%'.format(
            round(100*(value.achievement/baseline.achievement - 1), 1)))
        print('ROI:                 {}%'.format(round(value.roi - baseline.roi,2)))
        print('Affordability:       {}%'.format(
            round(100*(value.affordability/baseline.affordability - 1), 1)))
        # print('Affordable:          ${}'.format(round(value.affordable - baseline.affordable,1)))
        # 5/30/2022
        print('Unmet need:          ${}'.format(round(value.affordable - baseline.affordable,1)))
        print('Accessibility:       {}%'.format(
            round(100*(value.accessible/baseline.accessible - 1), 1)))


        # print('ACT:                 {}%'.format(
        #     round(100*(value.act_composite_high/baseline.act_composite_high - 1), 1)))
        # print('GPA:                 {}%'.format(
        #     round(100*(value.high_school_gpa/baseline.high_school_gpa - 1), 1)))
        # print('Need index:          {}%'.format(
        #     round(100*(1-value.need_index/baseline.need_index), 1)))

    #--------------------------------------------------------------------------
    def compare_against_baseline_per_category(self):
        def my_format(prefix=''):
            def f(x):
                if numpy.isnan(x):
                    return 'no-baseline'
                return '{:.1f}%'.format(x)
            return f 
        cat = ['revenue','enrollment','achievement_index','accessible','amount_mn','merit_cat','need_cat']
        stats = self.stats[cat].copy()
        # stats['merit_cat'].replace({
        #     i:'${}'.format(int(self.merit_buckets[i])) for i in range(len(self.merit_buckets))
        #     }, inplace=True)
        # stats['need_cat'].replace({
        #     i:'${}'.format(int(self.need_buckets[i])) for i in range(len(self.need_buckets))
        #     }, inplace=True)
        stats['merit_cat'].replace({
            i : int(self.merit_buckets[i]) for i in range(len(self.merit_buckets))
            }, inplace=True)
        stats['need_cat'].replace({
            i : int(self.need_buckets[i]) for i in range(len(self.need_buckets))
            }, inplace=True)

        output = stats.groupby(['merit_cat','need_cat']).agg(self.expected_value).round(1)
        output = output.style.format(formatter=dict(
            revenue = my_format('$'),
            enrollment = my_format(),
            achievement_index = my_format(),
            accessible = my_format(),
            amount_mn = my_format('$'),
            )).background_gradient(axis=0)

        return output


    #--------------------------------------------------------------------------
    def expected_value(self, r):
        # return self.compute_value(r, self.y_prob, r.index)
        # print('\nexpected_value', type(r), type(self.y_prob), type(self.y))

        current = self.compute_value(r, self.y_prob, r.index)
        baseline = self.compute_value(self.original_full_stats[r.name], self.y, r.index)
        if baseline==0:
            if current==0:
                return 0
            return numpy.nan
        return 100*(current/baseline - 1)

    #--------------------------------------------------------------------------

    def info(self, which):
        if which=='need':
            tmp = self.stats[['need_index','need_cat']].copy()
            tmp['est_annual_ft_pell'] = self.X.est_annual_ft_pell
            tmp = tmp.groupby('need_cat').agg('mean')

            need_info = pandas.cut(self.stats.need_index, self.need_quantiles,include_lowest=True).value_counts(1).sort_index()
            need_info = pandas.DataFrame(
                index=need_info.index, 
                columns=['percentage','amount'],
                data=list(zip((100*need_info.values).round(1),self.need_buckets)))
            need_info['mean_index'] = tmp[['need_index']].round(0).values
            need_info['mean_pell'] = tmp[['est_annual_ft_pell']].round(0).values
            need_info = need_info.style.format(formatter=dict(
                percentage='{:.1f}%', 
                amount='${:.1f}',
                mean_pell='${:.1f}',
                mean_index='{:.1f}',
                ))
            return need_info

        if which=='merit':
            tmp = self.stats[['achievement_index', 'merit_cat']].copy()
            tmp['amount_merit'] = self.X.amount_merit
            tmp['act_composite_high'] = self.X.act_composite_high
            tmp['high_school_gpa'] = self.X.high_school_gpa
            tmp = tmp.groupby('merit_cat').agg('mean')

            merit_info = pandas.qcut(self.stats.achievement_index, q=self.merit_quantiles).value_counts(1).sort_index()
            merit_info = pandas.DataFrame(
                index=merit_info.index, 
                columns=['percentage','amount'],
                data=list(zip((100*merit_info.values).round(1),self.merit_buckets)))
            merit_info['mean_index'] = tmp[['achievement_index']].round(1).values
            merit_info['mean_act'] = tmp[['act_composite_high']].round(1).values
            merit_info['mean_gpa'] = tmp[['high_school_gpa']].round(2).values
            merit_info = merit_info.style.format(formatter=dict(
                percentage='{}%',
                amount='${}',
                mean_index='{:.1f}',
                mean_act='{:.1f}',
                mean_gpa='{:.2f}'))
            return merit_info


    