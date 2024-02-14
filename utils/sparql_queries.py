######################################
#           Static queries           #
######################################

find_all_nodes_q = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?labeledEntity WHERE {
        {
            ?subject ?predicate ?object .
            ?predicate rdfs:label ?predicateLabel .
            ?subject rdfs:label ?subjectLabel.

            BIND(?subjectLabel AS ?labeledEntity) 
        }
        UNION
        {
            ?subject ?predicate ?object .
            ?predicate rdfs:label ?predicateLabel .
            ?object rdfs:label ?objectLabel.

            BIND(?objectLabel AS ?labeledEntity)
        }
    }"""

find_all_triples_q = """
    SELECT ?subject ?predicate ?object WHERE {
        ?subjectNode ?predicateNode ?objectNode .
        ?subjectNode rdfs:label ?subject .
        ?predicateNode rdfs:label ?predicate .
        ?objectNode rdfs:label ?object
    }"""

find_all_inheritances_q = """
    SELECT ?parent ?child WHERE {
        ?childNode rdf:type ?parentNode .
        ?parentNode rdfs:label ?parent .
        ?childNode rdfs:label ?child
    }"""

#####################################
#          Dynamic queries          #
#####################################

find_all_triples_with_node_q = """
    SELECT ?subject ?predicate ?object WHERE {{
        {{
            ?subjectNode rdfs:label "{0}" .
        }}
        UNION
        {{
            ?objectNode rdfs:label "{0}" .
        }}
        ?subjectNode ?predicateNode ?objectNode .
        ?subjectNode rdfs:label ?subject .
        ?predicateNode rdfs:label ?predicate .
        ?objectNode rdfs:label ?object .
    }}"""

find_all_triples_with_2_nodes_q = """
    SELECT DISTINCT ?subject ?predicate ?object WHERE {{
        {{
            {{
                ?subjectNode rdfs:label "{0}" .
            }}
            UNION
            {{
                ?objectNode rdfs:label "{0}" .
            }}
        }}
        UNION
        {{
            {{
                ?subjectNode rdfs:label "{1}" .
            }}
            UNION
            {{
                ?objectNode rdfs:label "{1}" .
            }}
        }}
        ?subjectNode ?predicateNode ?objectNode .
        ?subjectNode rdfs:label ?subject .
        ?predicateNode rdfs:label ?predicate .
        ?objectNode rdfs:label ?object .
    }}"""

find_all_edges_linking_nodes_directional_q = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?predicate_label WHERE {{
        ?subject rdfs:label "{0}" .
        ?subject ?predicate ?object .
        ?object rdfs:label "{1}" .
        ?predicate rdfs:label ?predicate_label .
    }}"""
