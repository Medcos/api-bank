## Afficher le fichier d'analyse drift
@app.route('/drift', methods=['GET'])
def drift():
    return render_template('drift.html')

if __name__ == '__main__':
    app.run()