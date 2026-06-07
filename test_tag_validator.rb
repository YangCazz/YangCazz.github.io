# Bypasses github-pages safe mode to load the tag validator plugin
require 'jekyll'

# Remove github-pages safe-mode restriction
Jekyll::Hooks.register :site, :after_reset do |site|
  plugin_path = File.expand_path('_plugins', site.source)
  if File.directory?(plugin_path)
    Dir[File.join(plugin_path, '*.rb')].each { |f| load f }
  end
end

conf = Jekyll.configuration(source: Dir.pwd)
site = Jekyll::Site.new(conf)
site.reset
site.read
