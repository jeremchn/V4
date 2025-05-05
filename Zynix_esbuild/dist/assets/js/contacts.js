// Fonction pour récupérer les paramètres de l'URL
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

// Charger et afficher les contacts
function loadContacts() {
    const companyName = getQueryParam('company'); // Récupérer le nom de l'entreprise depuis l'URL
    if (!companyName) {
        document.getElementById('company-title').textContent = 'Aucune entreprise sélectionnée';
        return;
    }

    // Mettre à jour le titre de la page
    document.getElementById('company-title').textContent = `Contacts pour ${companyName}`;

    // Charger le fichier CSV
    fetch('/contacts.csv')
        .then(response => response.text())
        .then(csvText => {
            // Utiliser PapaParse pour analyser le CSV
            const parsedData = Papa.parse(csvText, {
                header: true, // Utiliser la première ligne comme en-têtes
                skipEmptyLines: true // Ignorer les lignes vides
            });

            const contacts = parsedData.data; // Les données du CSV

            // Filtrer les contacts pour l'entreprise sélectionnée
            const filteredContacts = contacts.filter(contact => contact.Company === companyName);

            // Générer les lignes du tableau
            const tableBody = document.getElementById('contacts-table');
            tableBody.innerHTML = ''; // Vider le tableau avant d'ajouter les nouvelles lignes

            if (filteredContacts.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="5">Aucun contact trouvé pour cette entreprise.</td></tr>';
                return;
            }

            filteredContacts.forEach(contact => {
                const row = `
                    <tr>
                        <td>${contact['First Name']}</td>
                        <td>${contact['Last Name']}</td>
                        <td>${contact['Email']}</td>
                        <td>${contact['Position']}</td>
                        <td>${contact['Domain']}</td>
                    </tr>
                `;
                tableBody.innerHTML += row;
            });
        })
        .catch(error => {
            console.error('Erreur lors du chargement des contacts :', error);
            document.getElementById('contacts-table').innerHTML = '<tr><td colspan="5">Erreur lors du chargement des contacts.</td></tr>';
        });
}

// Charger les contacts au chargement de la page
document.addEventListener('DOMContentLoaded', loadContacts);