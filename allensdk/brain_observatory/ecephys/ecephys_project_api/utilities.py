import copy as cp

from jinja2 import Environment, BaseLoader, DictLoader


def postgres_macros():
    return {
        "postgres_macros": """"
            {% macro comma_sep(data, quote=False) %}
                ({% for datum in data%}
                    {% if quote%}\'{%endif -%}
                    {{datum}}
                    {%- if quote %}\'{% endif %}
                    {% if not loop.last %},{% endif %}
                {% endfor %})
            {% endmacro %}
            {% macro optional_contains(key, data, quote=False) %}
                {% if data is not none -%}
                    and {{key}} in {{comma_sep(data, quote)}}
                {% endif %}
            {% endmacro %}
            {% macro optional_equals(key, value) %}
                {% if data is not none -%}
                    and {{key}} = {{value}}
                {% endif %}
            {% endmacro %}
            {% macro optional_not_null(key, value=True) %}
                {% if value is not none -%}
                    and {{key}} is {{- ' not ' if value -}} null
                {% endif %}
            {% endmacro %}
            {% macro str(x) %}
                {{x ~ ""}}
            {% endmacro %}
        """
    }


def build_and_execute(query, base=None, engine=None, **kwargs):
    env = build_environment({"__tmp__": query}, base=base)
    return execute_templated(env, "__tmp__", engine=engine, **kwargs)


def build_environment(template_strings, base=None):
    if base is None:
        base = {}
    else:
        base = cp.deepcopy(base)

    base.update(template_strings)
    return Environment(loader=DictLoader(base), lstrip_blocks=True, trim_blocks=True)


def execute_templated(environment, name, engine, **kwargs):
    template = environment.get_template(name)
    rendered = template.render(**kwargs)
    return engine(rendered)
