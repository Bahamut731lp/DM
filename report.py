import pandas as pd
from tabulate import tabulate
from bs4 import BeautifulSoup

class Report:
    def __init__(self, file_path: str):
        self.model_results = []
        self.file = open(file_path, "w", encoding="utf-8")
        
        with open("./report/init.html", "r", encoding="utf-8") as template:
            self.content = template.read()

    def add_table(self, title, data: pd.DataFrame):
        html_table = tabulate(data, headers='keys', tablefmt="html")
        self.file.write(f"<h2>{title}</h2>\n")
        self.file.write(html_table)
        self.file.write("<hr>\n")

    def add_model_result(self, model: dict):
        rows = []

        data = model.get("stats")
        title = model.get("name")
        val_acc = model.get("validation", {}).get("acc", 0)
        test_acc = model.get("testing", {}).get("acc", 0)

        val_loss = model.get("validation", {}).get("loss")
        test_loss = model.get("testing", {}).get("loss")

        # Keep only rows with index starting with 'drug'
        drugs = data[data.index.str.startswith("drug")].copy()

        # Add new columns
        soup = BeautifulSoup(drugs.to_html(classes="min-w-full text-sm text-left", border=0), "html.parser")

        for th in soup.find_all("th"):
            th["class"] = "px-4 py-2 bg-gray-200"

        # Style table rows with striping
        for i, tr in enumerate(soup.find_all("tr")[1:]):  # Skip the header row
            tr["class"] = "odd:bg-white even:bg-gray-50"

            for td in tr.find_all("td"):
                td["class"] = "px-4 py-2"

        for index, item in data.iterrows():
            row = '<tr class="odd:bg-white even:bg-gray-50">'

            for key in data.keys():
                row += (f'<td class="px-4 py-2">{item[key]}</td>')

            row += '</tr>'
            rows.append(row)
        
        styled_table_html = str(soup)
        
        self.model_results.append(f"""
            <div class="bg-white shadow-md rounded-lg p-4 overflow-auto">
                <h3 class="text-xl font-bold mb-2">{title}
                    <span class="text-gray-600 text-sm">(val_acc: {val_acc:.3f}, test_acc: {test_acc:.3f})</span>
                </h3>
                {styled_table_html}
            </div>
            """)

    def end(self):
        self.content = self.content.replace(r"{{MODEL_RESULTS}}", "".join(self.model_results))

        self.file.write(self.content)
        self.file.close()
