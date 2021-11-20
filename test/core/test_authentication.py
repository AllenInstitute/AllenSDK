import pytest

from allensdk.core.authentication import (
    EnvCredentialProvider, credential_injector, set_credential_provider)


@pytest.mark.parametrize(
    "provider,credential_map,expected",
    [
        (EnvCredentialProvider({"LIMS_USER": "user", "LIMS_PASSWORD": "1234"}),
         {"user": "LIMS_USER", "password": "LIMS_PASSWORD"},
         ("user", "1234")),
    ]
)
def test_credential_injector(provider, credential_map, expected):
    def mock_func(*, user, password):
        return (user, password)
    assert (
        credential_injector(credential_map, provider)(mock_func)() == expected)


@pytest.mark.parametrize(
    "provider,credential_map,expected",
    [
        (EnvCredentialProvider({"LIMS_USER": "user", "LIMS_PASSWORD": "1234"}),
         {"user": "LIMS_USER"},
         ("user")),
    ]
)
def test_credential_injector_only_injects_existing_kwargs(
        provider, credential_map, expected):
    def mock_func(*, user):
        return user
    assert (
        credential_injector(credential_map, provider)(mock_func)() == expected)


@pytest.mark.parametrize(
    "provider,credential_map",
    [
        (EnvCredentialProvider({"LIMS_USER": "user", "LIMS_PASSWORD": "1234"}),
         {"user": "LIMS_USER", "password": "LIMS_PASSWORD"},),
    ]
)
def test_credential_injector_only_injects_mapped_credentials(
        provider, credential_map):
    def mock_func(*, user, db):
        pass
    with pytest.raises(TypeError):
        credential_injector(credential_map, provider)(mock_func)()


@pytest.mark.parametrize(
    "provider,credential_map",
    [
        (EnvCredentialProvider({"LIMS_USER": "user", "LIMS_PASSWORD": "1234"}),
         {"user": "LIMS_USER", "password": "LIMS_PASSWORD"},),
    ]
)
def test_credential_injector_preserves_function_args(provider, credential_map):
    def mock_func(arg1, kwarg1=None, *, user, password):
        return (arg1, kwarg1, user, password)
    assert (
        credential_injector(credential_map, provider)
        (mock_func)("arg1", kwarg1="kwarg1")
        == ("arg1", "kwarg1", "user", "1234"))


def test_credential_injector_with_provider_update():
    def mock_func(*, user):
        return user
    provider = EnvCredentialProvider({"LIMS_USER": "user"})
    credential_map = {"user": "LIMS_USER"}
    set_credential_provider(provider)
    assert credential_injector(credential_map)(mock_func)() == "user"
