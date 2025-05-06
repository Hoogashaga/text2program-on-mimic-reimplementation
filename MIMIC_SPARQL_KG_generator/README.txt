MIMIC-SPARQL Essential Files
===========================

This folder contains the minimal set of files needed to build the MIMIC-SPARQL knowledge graph.

File Structure:
- build_mimicsqlstar_db/: Contains scripts to build MIMICSQL* database from MIMICSQL database
- build_mimicsparql_kg/: Contains scripts to build MIMIC-SPARQL knowledge graph from MIMICSQL* database
- mimicsql/evaluation/mimic_db/: Contains the original MIMIC database file

Usage:

1. Build MIMICSQL* database:
   ```
   python build_mimicsqlstar_db/build_mimicstar_db_from_mimicsql_db.py
   ```
   This step will generate the mimicsqlstar.db file

2. Build MIMIC-SPARQL knowledge graph:
   - Complex mode (recommended):
     ```
     python build_mimicsparql_kg/build_complex_kg_from_mimicsqlstar_db.py
     ```
     This step will generate the mimic_sparqlstar_kg.xml file

   - Simple mode:
     ```
     python build_mimicsparql_kg/build_simple_kg_from_mimicsql_db.py
     ```
     This step will generate the mimic_sparql_kg.xml file

Note: Make sure your current working directory is the essential_files folder before executing these commands. 