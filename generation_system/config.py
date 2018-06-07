
constraints = {
    'needEv': [],
    'needList': [],
    'avoidEv': [],
    'avoidList':[]
 }

role_rc_structures = {
   'intransitive': [
       {'agent':'none'},
       {'agent':'transitive'},
       {'agent':'intransitive'}
      ],
   'transitive': [
       {'agent':'none','patient':'none'},
       {'agent':'none','patient':'transitive'},
       {'agent':'none','patient':'intransitive'},
       {'agent':'transitive','patient':'none'},
       {'agent':'transitive','patient':'transitive'},
       {'agent':'transitive','patient':'intransitive'},
       {'agent':'intransitive','patient':'none'},
       {'agent':'intransitive','patient':'transitive'},
       {'agent':'intransitive','patient':'intransitive'},
      ]
  }
