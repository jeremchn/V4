import pandas as pd

# Charger le DataFrame
df = pd.read_csv("companies_enriched.csv")

# Trier les entreprises par Total Score et sélectionner les 1000 meilleures
top_companies = df.sort_values(by="Total Score", ascending=False).head(1000)

# Générer les lignes du tableau HTML avec les styles
table_rows = ""
for _, row in top_companies.iterrows():
    company_name = row['Company Name']
    domain = row['Domain']
    table_rows += f"""
        <tr>
            <th scope="row"><input class="form-check-input" type="checkbox" value="" aria-label="..."></th>
            <td>
                <div class="d-flex align-items-center">
                    <span class="avatar avatar-xs me-2 online avatar-rounded">
                        <img src="../assets/images/faces/3.jpg" alt="img">
                    </span>{company_name}
                </div>
            </td>
            <td>{row['Industry']}</td>
            <td><span class="badge bg-info-transparent">{row['News Score']}</span></td>
            <td><span class="badge bg-primary-transparent">Actif</span></td>
            <td><a href="http://{domain}" target="_blank">{domain}</a></td>
            <td>
                <div class="progress progress-xs">
                    <div class="progress-bar bg-primary" role="progressbar" style="width: {row['Total Score %']}%" aria-valuenow="{row['Total Score %']}" aria-valuemin="0" aria-valuemax="100">
                    </div>
                </div>
            </td>
            <td>${row['2024 Revenue (USD)']:,}</td>
            <td>
                <div class="hstack gap-2 fs-15">
                    <a href="profile(1).html?company={company_name}" class="btn btn-icon btn-sm btn-info">
                        <i class="ri-edit-line"></i> Fiche Entreprise
                    </a>
                    <a href="contacts.html?company={company_name}" class="btn btn-icon btn-sm btn-warning">
                        <i class="ri-user-line"></i> Contacts
                    </a>
                </div>
            </td>
        </tr>
    """

# Chemin vers le fichier HTML
html_file_path = "/Users/jeremy/Desktop/V1/Zynix_esbuild/dist/html/companies-tables.html"

# Lire le fichier HTML
with open(html_file_path, "r", encoding="utf-8") as file:
    html_content = file.read()

# Remplacer le contenu du tableau existant
start_marker = "<!-- Start::table-data -->"
end_marker = "<!-- End::table-data -->"
start_index = html_content.find(start_marker) + len(start_marker)
end_index = html_content.find(end_marker)

if start_index != -1 and end_index != -1:
    updated_html_content = (
        html_content[:start_index]
        + table_rows
        + html_content[end_index:]
    )

    # Écrire les modifications dans le fichier HTML
    with open(html_file_path, "w", encoding="utf-8") as file:
        file.write(updated_html_content)

    print("Le tableau HTML a été mis à jour avec les nouvelles données.")
else:
    print("Les marqueurs de tableau n'ont pas été trouvés dans le fichier HTML.")