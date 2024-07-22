#!/usr/bin/env python3
from returns.context import Reader, ReaderIOResult, ReaderIOResultE, ReaderFutureResult, ReaderFutureResultE, ReaderResult, ReaderResultE
from typing import Optional
from returns.maybe import Maybe, maybe
from returns.io import IOResult, impure_safe, IO, IOSuccess, IOFailure
from pydantic import BaseModel
from returns.result import Result, safe
from returns.pipeline import flow
from returns.pointfree import bind

class UserProfile:
    def __init__(self, first_name):
        self.first_name = first_name

class UserCatalogue:
    def __init__(self):
        self.index = {}
    def add(self, first_name: str):
        self.index[first_name] = UserProfile(first_name)
        return "Success"
    def get(self, first_name: str):
        return self.index[first_name] 
                

users = UserCatalogue().add("Karolus")

def bindable(user) -> IOResult['UserProfile', None]:
    if user:
        return IOSuccess(user)
    return IOFailure(None)
    
def fetch_user_profile(user_id: int) -> Result['UserProfile', Exception]:
    """Fetches `UserProfile` TypedDict from foreign API."""
    return flow(
        user_id,           # initial value
        _get_user,         # get the user from array
        bind(_validate),   # validate the user
        _print,            # show the result to console
    )

@impure_safe
def _get_user(first_name: str) -> UserProfile:
    user = users[first_name]
    return user

@safe
def _validate(user: UserProfile) -> 'UserProfile':
    return user

@impure_safe
def _print(user) -> IOResult[None, Exception]:
    print(user)
    return None

result = fetch_user_profile(1).value_or("Couldn't find the user!")
result_str = result.bind(bindable)

print(type(result)) # -- returns.io.IO
print(type(result_str)) # -- returns.io.IOFailure
print(type(result.bind(str))) # -- str
print(type(result.bind(bindable))) # -- returns.io.IOFailure

print(
    f"We have the user result: {result.bind(str)}"
)

print(result_str)
