# Run after Jekyll build: bundle exec ruby _scripts/check-series-relations.rb . _site
require 'yaml'
require 'date'
require 'json'
require 'uri'
require 'nokogiri'

source, destination = ARGV.map { |path| File.expand_path(path) }
abort 'Usage: ruby _scripts/check-series-relations.rb SOURCE BUILT_SITE' unless source && destination

def check(value, message)
  raise message unless value
end

series = YAML.load_file(File.join(source, '_data/series.yml'))
posts = Dir[File.join(source, '_posts/*.md')].to_h do |path|
  front = YAML.safe_load(File.read(path).split(/^---\s*$\n?/, 3)[1], permitted_classes: [Date, Time])
  [File.basename(path, '.md').sub(/^\d{4}-\d{2}-\d{2}-/, ''), front]
end
pages = {}
Dir[File.join(destination, '**/*.html')].each do |path|
  doc = Nokogiri::HTML(File.read(path))
  schema = doc.css('script[type="application/ld+json"]').map { |node| JSON.parse(node.text) }.find { |entry| entry['@type'] == 'BlogPosting' }
  next unless schema
  pages[schema.fetch('headline')] = {doc: doc, schema: schema, path: path}
end
check(pages.size == posts.size, 'Some source posts have no built article page')
agent = series.find { |item| item['id'] == 'agent-systems' }
slugs = %w[harness-engineering-agent-kernel subagent-architecture-isolation agent-context-effective-state super-agent-continuous-event-driven-architecture]
check(agent.fetch('posts').map { |item| item['slug'] } == slugs, 'Agent series order changed or contains unpublished posts')
check(series.select { |item| item['relation_map'] }.map { |item| item['id'] } == ['agent-systems'], 'Reading map leaked into unrelated series')
series.each do |item|
  entries = item.fetch('posts')
  entries.each do |entry|
    check(posts.key?(entry['slug']), "Unpublished series target: #{entry['slug']}")
  end
  entries.each_with_index do |entry, index|
    doc = pages.fetch(posts.fetch(entry['slug']).fetch('title')).fetch(:doc)
    banner = doc.at_css('.series-banner')
    check(banner, 'Missing agent series banner') if item['id'] == 'agent-systems'
    next unless banner # Do not require new banners on legacy posts.
    check(banner.css('.series-banner-post').size == entries.size, "Wrong series count: #{entry['slug']}")
    check(banner.at_css('.series-banner-index').text.include?("第 #{index + 1} 篇 / 共 #{entries.size} 篇"), "Wrong series index: #{entry['slug']}")
    relations = banner.at_css('details.series-relations')
    unless item['id'] == 'agent-systems'
      check(relations.nil?, 'Unexpected relation map on legacy series')
      next
    end
    check(relations && relations.attribute('open').nil?, 'Reading map must be present and initially collapsed')
    check(relations.at_css('summary').text == '系列关联与阅读路径', 'Missing accessible map summary')
    rows = relations.css('tbody tr')
    check(rows.size == 4, 'Reading map must contain exactly the four published posts')
    rows.each_with_index do |row, row_index|
      target = pages.fetch(posts.fetch(slugs[row_index]).fetch('title')).fetch(:doc)
      canonical = URI(target.at_css('link[rel="canonical"]')['href']).path
      check(row.at_css('th a')['href'] == canonical, 'Reading map URL does not match canonical URL')
      check(row.css('td').size == 3 && row.css('td').all? { |cell| !cell.text.strip.empty? }, 'Missing relationship information')
    end
    check(relations.css('a[aria-current="page"]').size == 1, 'Current article must be marked once')
    links = doc.css('.series-nav-link')
    check(links.size == (index == 0 || index == entries.size - 1 ? 1 : 2), 'Incorrect previous/next boundaries')
    {'.prev' => index - 1, '.next' => index + 1}.each do |selector, adjacent|
      next unless adjacent.between?(0, entries.size - 1)
      target = pages.fetch(posts.fetch(slugs[adjacent]).fetch('title')).fetch(:doc)
      check(doc.at_css(".series-nav-link#{selector}")['href'] == URI(target.at_css('link[rel="canonical"]')['href']).path, 'Incorrect previous/next target')
    end
  end
end
context = posts.fetch(slugs[2])
super_agent = posts.fetch(slugs[3])
check(DateTime.parse(context.fetch('date').to_s) < DateTime.parse(super_agent.fetch('date').to_s), 'Context must be published before Super Agent')
check(context['permalink'] == '/2026/11/01/agent-context-effective-state/', 'Original context URL must remain stable')
context_page = pages.fetch(context.fetch('title'))
check(context_page[:schema]['datePublished'] == '2026-10-31', 'Published schema date was not corrected')
check(context_page[:doc].at_css('.post-date').text == '2026年10月31日', 'Visible publication date was not corrected')
check(context_page[:doc].at_css('link[rel="canonical"]')['href'] == 'https://yangcazz.github.io/2026/11/01/agent-context-effective-state/', 'Context canonical URL changed')
check(!File.exist?(File.join(destination, '2026/10/31/agent-context-effective-state/index.html')), 'Duplicate context URL generated')
puts "Series checks passed: #{series.size} series; four agent reading maps; #{pages.size} article pages; stable context URL and corrected date."
