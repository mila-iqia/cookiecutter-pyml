{% set is_open_source = cookiecutter.open_source_license != 'Not open source' -%}
{% for _ in cookiecutter.project_name %}={% endfor %}
{{ cookiecutter.project_name }}
{% for _ in cookiecutter.project_name %}={% endfor %}


{{ cookiecutter.project_short_description }}

{% if is_open_source %}
* Free software: {{ cookiecutter.open_source_license }}
{% endif %}


Instructions to setup the project_name
--------

To setup pre-commit hooks (to use flake8 before avery commit):

    cd .git/hooks/ && ln -s ../../config/hooks/pre-commit . && cd -


Features
--------

* __TODO__

