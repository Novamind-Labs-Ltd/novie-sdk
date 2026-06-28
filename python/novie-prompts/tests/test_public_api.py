def test_public_surface_is_importable():
    import novie_prompts

    assert callable(novie_prompts.get_managed_prompt)
    assert callable(novie_prompts.configure)
    assert callable(novie_prompts.set_recorder)
    assert callable(novie_prompts.has_recorder)
