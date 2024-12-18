import httpx
import pandas as pd


def run_sparql_query(
    query, endpoint="https://minmod.isi.edu/sparql", values=False, csv_path=None
):
    # add prefixes
    final_query = (
        """
    PREFIX : <https://minmod.isi.edu/ontology/>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX gkbi: <https://geokb.wikibase.cloud/entity/>
    PREFIX gkbt: <https://geokb.wikibase.cloud/prop/direct/>
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>
    \n"""
        + query
    )

    response = httpx.post(
        url=endpoint,
        data={"query": final_query},
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/sparql-results+json",
        },
        verify=False,
        # Need this for httpx to override the default timeout of 5 seconds
        timeout=360,
    )

    # print(response.text)

    try:
        qres = response.json()
        if "results" in qres and "bindings" in qres["results"]:
            df = pd.json_normalize(qres["results"]["bindings"])
            if values:
                filtered_columns = df.filter(like=".value").columns
                df = df[filtered_columns]
            if csv_path:
                df.to_csv(csv_path, index=False)
            return df
    except Exception:
        return None


def run_minmod_query(query, values=False, csv_path=None):
    return run_sparql_query(
        query,
        endpoint="https://minmod.isi.edu/sparql",
        values=values,
        csv_path=csv_path,
    )
