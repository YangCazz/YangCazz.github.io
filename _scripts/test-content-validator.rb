# Run: bundle exec ruby _scripts/test-content-validator.rb
require 'minitest/autorun'
require 'jekyll'
require 'yaml'
require_relative '../_plugins/content_validator'

class ContentValidatorTest < Minitest::Test
  Validator = BlogContentValidator

  def test_html_references_resolve_citations
    raw = '<p>See <cite>[1]</cite> and <cite>[2]</cite>.</p><ol class="references"><li>First</li><li>Second</li></ol>'
    assert_empty Validator.missing_citations(raw)
  end

  def test_missing_html_reference_still_fails
    assert_equal [2], Validator.missing_citations('<cite>[2]</cite><ol class="references"><li>First</li></ol>')
  end

  def test_html_reference_start_and_explicit_values
    raw = '<cite>[3]</cite><cite>[7]</cite><cite>[8]</cite><ol class="references" start="3"><li>Third</li><li value="7">Seventh</li><li>Eighth</li></ol>'
    assert_empty Validator.missing_citations(raw)
  end

  def test_nested_list_items_are_not_extra_references
    raw = '<cite>[2]</cite><ol class="references"><li>First<ul><li>A detail</li></ul></li></ol>'
    assert_equal [2], Validator.missing_citations(raw)
  end

  def test_markdown_references_use_rendered_list_numbering
    raw = "See <cite>[2]</cite>.\n\n## 参考文献\n\n1. First\n1. Second\n"
    assert_empty Validator.missing_citations(raw)
  end

  def test_unrelated_numbered_steps_cannot_mask_a_missing_reference
    raw = "<cite>[2]</cite>\n\n1. First step\n2. Second step\n\n## 参考文献\n\n1. Only reference\n"
    assert_equal [2], Validator.missing_citations(raw)
  end

  def test_reference_section_ends_at_next_same_level_heading
    raw = "<cite>[2]</cite>\n\n## References\n\n1. Only reference\n\n## Appendix\n\n1. Step one\n2. Step two\n"
    assert_equal [2], Validator.missing_citations(raw)
  end

  def test_fenced_examples_and_comments_are_not_citations
    raw = "````html\n```\n<cite>[99]</cite>\n```\n````\n<!-- <cite>[98]</cite> -->\n"
    assert_empty Validator.missing_citations(raw)
  end

  def test_fenced_reference_example_does_not_satisfy_real_citation
    raw = "<cite>[1]</cite>\n\n~~~html\n<ol class=\"references\"><li>Example only</li></ol>\n~~~\n"
    assert_equal [1], Validator.missing_citations(raw)
  end

  def test_registry_rejects_ambiguous_alias_instead_of_last_write_wins
    registry = [tag('First', aliases: ['Shared']), tag('Second', aliases: ['Shared'])]
    errors = Validator.registry_index(registry).last
    assert errors.any? { |error| error.include?("maps to both 'First' and 'Second'") }
  end

  def test_registry_rejects_alias_that_is_another_canonical_name
    errors = Validator.registry_index([tag('First', aliases: ['Second']), tag('Second')]).last
    assert errors.any? { |error| error.include?('conflicts with a canonical tag') }
  end

  def test_registry_rejects_duplicate_canonical_names_and_invalid_types
    errors = Validator.registry_index([tag('First'), tag('First', type: 'unknown')]).last
    assert errors.any? { |error| error.include?('duplicate canonical') }
    assert errors.any? { |error| error.include?('invalid type') }
  end

  def test_real_registry_has_no_conflicts_and_preserves_specific_methods
    canonical, types, aliases, errors = Validator.registry_index(YAML.load_file(File.expand_path('../_data/tags.yml', __dir__)))
    assert_empty errors
    %w[GraphRAG BM25 RRF ChromaDB SSE 模块化].each do |name|
      assert_includes canonical, name
      assert_equal 'method', types[name]
      refute aliases.key?(name)
    end
    ['认知架构', 'Cognitive Architecture', 'CoALA'].each { |name| refute aliases.key?(name) }
  end

  def test_all_existing_posts_have_resolvable_citations_and_canonical_tags
    registry = YAML.load_file(File.expand_path('../_data/tags.yml', __dir__))
    canonical = Validator.registry_index(registry).first
    Dir[File.expand_path('../_posts/*.md', __dir__)].each do |path|
      _, front, body = File.read(path).split(/^---\s*$\n?/, 3)
      data = YAML.safe_load(front, permitted_classes: [Date, Time])
      assert_empty Validator.missing_citations(body), File.basename(path)
      assert_empty Array(data['tags']) - canonical.to_a, File.basename(path)
      assert_equal Array(data['tags']).uniq, data['tags'], File.basename(path)
    end
  end

  def test_missing_citation_is_fatal_for_new_posts
    site, logger = fixture_site(Date.new(2026, 8, 2), '<cite>[2]</cite><ol class="references"><li>Only one</li></ol>')
    error = assert_raises(RuntimeError) { Jekyll.stub(:logger, logger) { Jekyll::Hooks.trigger(:site, :post_read, site) } }
    assert_includes error.message, 'Content validation failed'
    assert logger.errors.flatten.join.include?('cite references 2')
  end

  def test_missing_citation_remains_a_warning_for_legacy_posts
    site, logger = fixture_site(Date.new(2026, 8, 1), '<cite>[2]</cite><ol class="references"><li>Only one</li></ol>')
    Jekyll.stub(:logger, logger) { Jekyll::Hooks.trigger(:site, :post_read, site) }
    assert_empty logger.errors
    assert logger.warnings.flatten.join.include?('cite references 2')
    assert logger.warnings.flatten.join.include?('[legacy]')
  end

  def test_ambiguous_registry_is_fatal_even_when_only_legacy_posts_use_it
    site, logger = fixture_site(Date.new(2026, 8, 1), 'No citations')
    site.data['tags'][0]['aliases'] = ['Shared']
    site.data['tags'][1]['aliases'] = ['Shared']
    assert_raises(RuntimeError) { Jekyll.stub(:logger, logger) { Jekyll::Hooks.trigger(:site, :post_read, site) } }
    assert logger.errors.flatten.join.include?("alias 'Shared' maps to both")
  end

  private

  def tag(name, type: 'method', aliases: [])
    { 'name' => name, 'type' => type, 'aliases' => aliases }
  end

  def fixture_site(date, content)
    timestamp = Time.new(date.year, date.month, date.day, 10, 0, 0, '+08:00')
    data = { 'title' => 'Test', 'date' => timestamp, 'categories' => ['Test'], 'tags' => ['Domain', 'Method'], 'excerpt' => 'Test excerpt' }
    post = Struct.new(:date, :relative_path, :content, :data).new(timestamp, '_posts/test.md', content, data)
    site = Struct.new(:data, :posts).new({ 'tags' => [tag('Domain', type: 'domain'), tag('Method')] }, Struct.new(:docs).new([post]))
    logger = Class.new do
      attr_reader :errors, :warnings
      def initialize; @errors = []; @warnings = []; end
      def error(*messages); @errors << messages; end
      def warn(*messages); @warnings << messages; end
      def info(*); end
    end.new
    [site, logger]
  end
end
