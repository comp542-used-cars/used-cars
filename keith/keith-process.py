from typing import Iterator
import pandas as pd
import matplotlib.pyplot as plt
import io

class project3(object):
        
    def get_data_sets(self)-> Iterator[pd.DataFrame]:
        yield(pd.read_csv("C:\\Users\\keith\\My Drive\\School\\CSUN\\COMP542\\Projects\\Data\\Raw\\vehicles.csv",low_memory=False))
        yield(pd.read_csv("C:\\Users\\keith\\My Drive\\School\\CSUN\\COMP542\\Projects\\Data\\Raw\\used_cars_data.csv",low_memory=False))
    
    def get_names(self) -> Iterator[str]:
        yield('vehicles')
        yield('used_cars_data')

    def get_df_data(self,df : pd.DataFrame, name : str):
        folder = f"data/tables"
        df.describe(include='all').to_csv(f"{folder}/{name}_describe.csv")
        df.corr(method='pearson',numeric_only=True).to_csv(f"{folder}/{name}_correlation.csv")
        df.skew().to_csv(f"{folder}/{name}_skew.csv")
        buffer = io.StringIO()
        df.info(verbose=True,buf=buffer)
        s = buffer.getvalue()
        with open(f"{folder}/{name}_info.csv","w+") as f:
            f.write(s)
            
    def combine_sets(self,dfs):
        vehicles_manufacturer = dfs[0].groupby(["manufacturer"])["price"].median()
        used_manufacturer = dfs[1]
        used_manufacturer["manufacturer"] = used_manufacturer["make_name"]
        used_manufacturer = used_manufacturer.groupby(["manufacturer"])["price"].median()
        both_manufacturer = pd.concat([vehicles_manufacturer, used_manufacturer])
        print(both_manufacturer)
        
    def create_plots(self, df: pd.DataFrame):
        folder = f"data/graphs"
        aggregation_functions = {'price': 'median', 'manufacturer': 'first'}
        df_new = df.groupby(df['manufacturer']).aggregate(aggregation_functions)
        fig = df_new.plot(x="manufacturer",y="price", kind = 'bar')
        plt.show()
        
    def create_manufacture_plots(self, dfs):
        folder = f"data/graphs"
        vehicles_manufacturer = dfs[0]
        #vehicles_manufacturer = dfs[0].groupby(["manufacturer"])["price"]
        used_manufacturer = dfs[1]
        used_manufacturer["manufacturer"] = used_manufacturer["make_name"].str.lower()
        vehicles_manufacturer["manufacturer"] = vehicles_manufacturer["manufacturer"].str.lower()
        #used_manufacturer = used_manufacturer.groupby(["manufacturer"])["price"]
        both_manufacturer = pd.concat([vehicles_manufacturer, used_manufacturer])
        both_manufacturer_filter = both_manufacturer[both_manufacturer['price'] < 150000]
        
        both_manufacturer_filter_pivot = both_manufacturer_filter.pivot_table(index='vin',columns='manufacturer', values='price')
        #print(both_manufacturer_filter_pivot.head())
        fig = both_manufacturer_filter_pivot.boxplot(xlabel='Price',ylabel='Manufacturer',vert=False,figsize=(11,20),showfliers=False)
        #fig = both_manufacturer_filter_pivot.plot(kind='box',title='Year vs Price Box Plot',x='manufacturer',y='price',xlabel='Manufacturer',ylabel='Price',figsize=(80,20))
        #fig = both_manufacturer_filter_pivot.plot(x="manufacturer",y="price",xlabel='Manufacturer',ylabel='Price',kind = 'box',figsize=(40,11),title='Manufacturer vs Price Box Plot',showmeans=True)
        plt.savefig(f"{folder}/manufacturer_price_box_plot.png")
        print("Created manufacturer_price_box_plot.png")
        plt.clf()
        plt.close()
        
        fig = both_manufacturer['manufacturer'].value_counts()[:20].plot(kind='pie',title='Top 20 Manufacturers',figsize=(11,6))
        plt.savefig(f"{folder}/manufacturer_pie_plot.png")
        plt.clf()
        plt.close()
        print("Created manufacturer_pie_plot.png")
        
        fig = both_manufacturer['year'].value_counts().plot(kind='pie',title='Top Years',figsize=(8,4.8))
        plt.savefig(f"{folder}/year_pie_plot.png")
        plt.clf()
        plt.close()
        print("Created year_pie_plot.png")
        
        both_manufacturer_filter_pivot_year = both_manufacturer_filter.pivot_table(index='vin',columns='year', values='price')
        fig = both_manufacturer_filter_pivot_year.boxplot(xlabel='Price',ylabel='Year',vert=False, figsize=(10,50),showfliers=False)
        #fig = both_manufacturer_filter_pivot_year.plot(x='year',y='price',kind='box',title='Year vs Price Box Plot',xlabel='Year',ylabel='Price',figsize=(80,11))
        plt.savefig(f"{folder}/year_price_box_plot.png")
        plt.clf()
        plt.close()
        print("Created year_price_box_plot.png")
        
        both_manufacturer_filter_pivot_year_median = both_manufacturer_filter_pivot_year.median()
        fig = both_manufacturer_filter_pivot_year_median.plot(x="year",y="price",xlabel='Year',ylabel='Price',kind = 'line',figsize=(40,9),title='Year vs Median Price Line Plot')
        plt.savefig(f"{folder}/year_median_price_line_plot.png")
        plt.clf()
        plt.close()   
        print("Created year_median_line_plot.png")
        
        
        vehicles_manufacturer_filter = vehicles_manufacturer[vehicles_manufacturer['price'] < 150000]
        vehicles_manufacturer_filter_pivot_state = vehicles_manufacturer_filter.pivot_table(index='VIN',columns='state', values='price')
        fig = vehicles_manufacturer_filter_pivot_state.boxplot(xlabel='Price',ylabel='State',vert=False,figsize=(10,20),showfliers=False)
        #fig = both_manufacturer_filter_pivot_state.plot(x="city",y="price",xlabel='City',ylabel='Price',kind = 'box',title='City vs Price Box Plot')
        plt.savefig(f"{folder}/state_price_box_plot.png")
        plt.clf()
        plt.close()
        print("Created state_price_box_plot.png")
        
        
        vehicles_manufacturer_pivot_state_median = vehicles_manufacturer_filter_pivot_state.median()
        fig = vehicles_manufacturer_pivot_state_median.plot(x="state",y="price",xlabel='State',ylabel='Price',kind = 'bar',title='State vs Median Price Bar Plot',figsize=(50,10))
        plt.savefig(f"{folder}/state_price_bar_plot.png")
        plt.clf()
        plt.close()
        print("Created state_price_bar_plot.png")
        
        
        

    def main(self):
        dfs = []
        for df,name in zip(self.get_data_sets(),self.get_names()):
            #self.get_df_data(df, name)
            dfs.append(df)
        self.create_manufacture_plots(dfs)
        
             
if __name__ == "__main__":
    controlller = project3()
    controlller.main()

